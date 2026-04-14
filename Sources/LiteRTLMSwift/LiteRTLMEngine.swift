import Foundation
import CoreGraphics
import ImageIO
import os
import CLiteRTLM

/// Swift wrapper for Google's LiteRT-LM on-device inference engine.
///
/// Supports text generation (Session API) and multimodal vision inference
/// (Conversation API) with `.litertlm` model files (e.g. Gemma 4 E2B).
///
/// Thread safety: all C API calls are serialized on an internal dispatch queue.
/// The class is `@unchecked Sendable` because OpaquePointers are only accessed
/// on that queue.
///
/// ## Quick Start
/// ```swift
/// let engine = LiteRTLMEngine(modelPath: modelURL)
/// try await engine.load()
///
/// // Text generation
/// let response = try await engine.generate(prompt: "Hello!", temperature: 0.7, maxTokens: 256)
///
/// // Vision
/// let caption = try await engine.vision(imageData: jpegData, prompt: "Describe this photo.", temperature: 0.7, maxTokens: 512)
/// ```
@Observable
public final class LiteRTLMEngine: @unchecked Sendable {

    // MARK: - Types

    public enum Status: Sendable, Equatable {
        case notLoaded
        case loading
        case ready
        case error(String)
    }

    // MARK: - Properties

    public private(set) var status: Status = .notLoaded

    /// Vision is available when the model is loaded — `.litertlm` embeds the vision encoder.
    public var isReady: Bool { status == .ready }

    private let modelPath: URL
    private let backend: String

    private var engine: OpaquePointer?  // LiteRtLmEngine*
    private let inferenceQueue = DispatchQueue(label: "com.litertlm.inference", qos: .userInitiated)

    private static let log = Logger(subsystem: "LiteRTLMSwift", category: "Engine")

    // MARK: - Init

    /// Create an engine instance.
    /// - Parameters:
    ///   - modelPath: Path to the `.litertlm` model file on disk.
    ///   - backend: Compute backend — `"cpu"` or `"gpu"` (GPU uses Metal on iOS).
    public init(modelPath: URL, backend: String = "cpu") {
        self.modelPath = modelPath
        self.backend = backend
    }

    deinit {
        let eng = engine
        let queue = inferenceQueue
        if eng != nil {
            queue.async {
                if let e = eng { litert_lm_engine_delete(e) }
            }
        }
    }

    // MARK: - Lifecycle

    /// Load the `.litertlm` model. Call once, reuse for multiple inferences.
    /// The vision encoder is embedded in the model file — no separate load step needed.
    @MainActor
    public func load() async throws {
        guard status != .ready else { return }

        status = .loading
        Self.log.info("Loading model: \(self.modelPath.lastPathComponent), backend: \(self.backend)")

        let path = modelPath.path
        let backendStr = self.backend
        let startTime = CFAbsoluteTimeGetCurrent()

        guard FileManager.default.fileExists(atPath: path) else {
            let msg = "Model file not found at \(path)"
            Self.log.error("\(msg)")
            status = .error(msg)
            throw LiteRTLMError.modelNotFound
        }

        do {
            let createdEngine = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<OpaquePointer, any Error>) in
                self.inferenceQueue.async {
                    do {
                        litert_lm_set_min_log_level(1)

                        guard let settings = litert_lm_engine_settings_create(
                            path, backendStr, backendStr, nil
                        ) else {
                            throw LiteRTLMError.engineCreationFailed("Failed to create engine settings")
                        }

                        litert_lm_engine_settings_set_max_num_tokens(settings, 4096)

                        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
                            .appendingPathComponent("litertlm_cache").path
                        try? FileManager.default.createDirectory(atPath: cacheDir, withIntermediateDirectories: true)
                        litert_lm_engine_settings_set_cache_dir(settings, cacheDir)

                        litert_lm_engine_settings_enable_benchmark(settings)

                        guard let createdEngine = litert_lm_engine_create(settings) else {
                            litert_lm_engine_settings_delete(settings)
                            throw LiteRTLMError.engineCreationFailed("litert_lm_engine_create returned NULL")
                        }
                        litert_lm_engine_settings_delete(settings)

                        continuation.resume(returning: createdEngine)
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            }

            inferenceQueue.sync { self.engine = createdEngine }

            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            Self.log.info("Model loaded in \(String(format: "%.1f", elapsed))s")
            status = .ready
        } catch {
            let msg = "Load failed: \(error.localizedDescription)"
            Self.log.error("\(msg)")
            status = .error(msg)
            throw error
        }
    }

    /// Unload the model to free memory.
    @MainActor
    public func unload() {
        inferenceQueue.sync {
            if let s = chatSession {
                litert_lm_session_delete(s)
                chatSession = nil
            }
            if let c = chatSessionConfig {
                litert_lm_session_config_delete(c)
                chatSessionConfig = nil
            }
            if let eng = engine { litert_lm_engine_delete(eng) }
            engine = nil
        }
        status = .notLoaded
        Self.log.info("Model unloaded")
    }

    // MARK: - Text Generation (Session API)

    /// Generate text from a prompt. Creates a one-shot session per call.
    ///
    /// - Parameters:
    ///   - prompt: The input text. For Gemma 4, use `<|turn>user\n...<turn|>\n<|turn>model\n` format.
    ///   - temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative). Default 0.7.
    ///   - maxTokens: Maximum tokens to generate. Default 512.
    /// - Returns: Generated text.
    public func generate(
        prompt: String,
        temperature: Float = 0.7,
        maxTokens: Int = 512
    ) async throws -> String {
        try ensureReady()
        return try await runSessionInference(
            prompt: prompt, temperature: temperature, maxTokens: Int32(maxTokens)
        )
    }

    /// Stream text generation token by token.
    ///
    /// Creates a one-shot session per call. For multi-turn conversations with
    /// KV cache reuse, use the persistent session API instead.
    ///
    /// - Parameters:
    ///   - prompt: The input text.
    ///   - temperature: Sampling temperature. Default 0.7.
    ///   - maxTokens: Maximum tokens to generate. Default 512.
    /// - Returns: An `AsyncThrowingStream` yielding text chunks.
    public func generateStreaming(
        prompt: String,
        temperature: Float = 0.7,
        maxTokens: Int = 512
    ) -> AsyncThrowingStream<String, Error> {
        runSessionInferenceStreaming(
            prompt: prompt, temperature: temperature, maxTokens: Int32(maxTokens)
        )
    }

    // MARK: - Vision (Conversation API)

    /// Run vision inference on a single image.
    ///
    /// Uses the Conversation API, which handles image decoding, resizing, and
    /// patchification internally. Input images are auto-converted to JPEG and
    /// resized to fit within `maxImageDimension`.
    ///
    /// - Parameters:
    ///   - imageData: Raw image bytes (JPEG, PNG, HEIC, etc.).
    ///   - prompt: Text prompt for the vision model (e.g., "Describe this photo.").
    ///   - temperature: Sampling temperature. Default 0.7.
    ///   - maxTokens: Maximum tokens to generate. Default 512.
    ///   - maxImageDimension: Resize long edge to this value. Default 1024.
    /// - Returns: Generated text response.
    public func vision(
        imageData: Data,
        prompt: String,
        temperature: Float = 0.7,
        maxTokens: Int = 512,
        maxImageDimension: Int = 1024
    ) async throws -> String {
        try ensureReady()

        guard let jpegData = Self.prepareImageForVision(imageData, maxDimension: maxImageDimension) else {
            throw LiteRTLMError.inferenceFailure("Failed to convert image to JPEG")
        }

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + ".jpg")
        try jpegData.write(to: tempURL)

        let result = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<String, any Error>) in
            self.inferenceQueue.async { [self] in
                defer { try? FileManager.default.removeItem(at: tempURL) }
                do {
                    guard let eng = self.engine else { throw LiteRTLMError.modelNotLoaded }

                    guard let sessionConfig = litert_lm_session_config_create() else {
                        throw LiteRTLMError.inferenceFailure("Failed to create session config")
                    }
                    litert_lm_session_config_set_max_output_tokens(sessionConfig, Int32(maxTokens))
                    var samplerParams = LiteRtLmSamplerParams(
                        type: kTopP, top_k: 40, top_p: 0.95,
                        temperature: temperature, seed: 0
                    )
                    litert_lm_session_config_set_sampler_params(sessionConfig, &samplerParams)

                    guard let convConfig = litert_lm_conversation_config_create(
                        eng, sessionConfig, nil, nil, nil, false
                    ) else {
                        litert_lm_session_config_delete(sessionConfig)
                        throw LiteRTLMError.inferenceFailure("Failed to create conversation config")
                    }

                    guard let conversation = litert_lm_conversation_create(eng, convConfig) else {
                        litert_lm_conversation_config_delete(convConfig)
                        litert_lm_session_config_delete(sessionConfig)
                        throw LiteRTLMError.inferenceFailure("Failed to create conversation")
                    }
                    defer {
                        litert_lm_conversation_delete(conversation)
                        litert_lm_conversation_config_delete(convConfig)
                        litert_lm_session_config_delete(sessionConfig)
                    }

                    let messageJSON = Self.buildVisionMessageJSON(
                        imagePath: tempURL.path, text: prompt
                    )

                    guard let response = messageJSON.withCString({ msgPtr in
                        litert_lm_conversation_send_message(conversation, msgPtr, nil)
                    }) else {
                        throw LiteRTLMError.inferenceFailure("Vision inference returned no response")
                    }
                    defer { litert_lm_json_response_delete(response) }

                    guard let responsePtr = litert_lm_json_response_get_string(response) else {
                        throw LiteRTLMError.inferenceFailure("Vision response string is NULL")
                    }

                    let result = Self.extractTextFromConversationResponse(String(cString: responsePtr))
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }

        return result
    }

    /// Run vision inference on multiple images.
    ///
    /// - Parameters:
    ///   - imagesData: Array of raw image bytes.
    ///   - prompt: Text prompt about the images.
    ///   - temperature: Sampling temperature. Default 0.7.
    ///   - maxTokens: Maximum tokens to generate. Default 1024.
    ///   - maxImageDimension: Resize long edge to this value. Default 1024.
    /// - Returns: Generated text response.
    public func visionMultiImage(
        imagesData: [Data],
        prompt: String,
        temperature: Float = 0.7,
        maxTokens: Int = 1024,
        maxImageDimension: Int = 1024
    ) async throws -> String {
        try ensureReady()
        guard !imagesData.isEmpty else {
            throw LiteRTLMError.inferenceFailure("No images provided")
        }

        var tempURLs: [URL] = []
        for (i, data) in imagesData.enumerated() {
            guard let jpegData = Self.prepareImageForVision(data, maxDimension: maxImageDimension) else {
                throw LiteRTLMError.inferenceFailure("Failed to convert image \(i + 1) to JPEG")
            }
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString + ".jpg")
            try jpegData.write(to: url)
            tempURLs.append(url)
        }

        let urlsToCleanup = tempURLs
        let result = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<String, any Error>) in
            self.inferenceQueue.async { [self, urlsToCleanup] in
                defer {
                    for url in urlsToCleanup {
                        try? FileManager.default.removeItem(at: url)
                    }
                }
                do {
                    guard let eng = self.engine else { throw LiteRTLMError.modelNotLoaded }

                    guard let sessionConfig = litert_lm_session_config_create() else {
                        throw LiteRTLMError.inferenceFailure("Failed to create session config")
                    }
                    litert_lm_session_config_set_max_output_tokens(sessionConfig, Int32(maxTokens))
                    var samplerParams = LiteRtLmSamplerParams(
                        type: kTopP, top_k: 40, top_p: 0.95,
                        temperature: temperature, seed: 0
                    )
                    litert_lm_session_config_set_sampler_params(sessionConfig, &samplerParams)

                    guard let convConfig = litert_lm_conversation_config_create(
                        eng, sessionConfig, nil, nil, nil, false
                    ) else {
                        litert_lm_session_config_delete(sessionConfig)
                        throw LiteRTLMError.inferenceFailure("Failed to create conversation config")
                    }

                    guard let conversation = litert_lm_conversation_create(eng, convConfig) else {
                        litert_lm_conversation_config_delete(convConfig)
                        litert_lm_session_config_delete(sessionConfig)
                        throw LiteRTLMError.inferenceFailure("Failed to create conversation")
                    }
                    defer {
                        litert_lm_conversation_delete(conversation)
                        litert_lm_conversation_config_delete(convConfig)
                        litert_lm_session_config_delete(sessionConfig)
                    }

                    let messageJSON = Self.buildMultiImageMessageJSON(
                        imagePaths: tempURLs.map(\.path), text: prompt
                    )

                    guard let response = messageJSON.withCString({ msgPtr in
                        litert_lm_conversation_send_message(conversation, msgPtr, nil)
                    }) else {
                        throw LiteRTLMError.inferenceFailure("Vision inference returned no response")
                    }
                    defer { litert_lm_json_response_delete(response) }

                    guard let responsePtr = litert_lm_json_response_get_string(response) else {
                        throw LiteRTLMError.inferenceFailure("Vision response string is NULL")
                    }

                    let result = Self.extractTextFromConversationResponse(String(cString: responsePtr))
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }

        return result
    }

    // MARK: - Persistent Session (KV Cache Reuse)
    //
    // LiteRT-LM's Session maintains a KV cache across multiple generate_content
    // calls. By keeping the session alive across turns, subsequent messages only
    // need to prefill NEW tokens instead of the entire conversation history.
    // This reduces TTFT from ~20s (full prefill) to ~1-2s (incremental).

    private var chatSession: OpaquePointer?
    private var chatSessionConfig: OpaquePointer?

    /// Open a persistent session for multi-turn generation with KV cache reuse.
    ///
    /// Call once when a conversation begins. Subsequent calls to
    /// `sessionGenerateStreaming(input:)` reuse this session's KV cache.
    ///
    /// - Parameters:
    ///   - temperature: Sampling temperature. Default 0.3.
    ///   - maxTokens: Maximum tokens per generation. Default 512.
    public func openSession(temperature: Float = 0.3, maxTokens: Int = 512) async throws {
        try ensureReady()
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            inferenceQueue.async { [self] in
                do {
                    if let s = chatSession {
                        litert_lm_session_delete(s)
                        chatSession = nil
                    }
                    if let c = chatSessionConfig {
                        litert_lm_session_config_delete(c)
                        chatSessionConfig = nil
                    }

                    guard let eng = engine else { throw LiteRTLMError.modelNotLoaded }
                    let (session, config) = try createSession(
                        engine: eng, temperature: temperature, maxTokens: Int32(maxTokens)
                    )
                    chatSession = session
                    chatSessionConfig = config
                    Self.log.info("Persistent session opened")
                    cont.resume()
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    /// Close the persistent session, freeing KV cache memory.
    public func closeSession() {
        inferenceQueue.async { [self] in
            guard chatSession != nil else { return }
            if let s = chatSession {
                logSessionBenchmark(s)
                litert_lm_session_delete(s)
                chatSession = nil
            }
            if let c = chatSessionConfig {
                litert_lm_session_config_delete(c)
                chatSessionConfig = nil
            }
            Self.log.info("Persistent session closed")
        }
    }

    /// Stream text using the persistent session.
    ///
    /// `input` should be ONLY the new turn content — the session's KV cache
    /// already holds all previous context.
    ///
    /// - Parameter input: New input text for this turn.
    /// - Returns: An `AsyncThrowingStream` yielding text chunks.
    public func sessionGenerateStreaming(input: String) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            self.inferenceQueue.async { [self] in
                guard let session = self.chatSession else {
                    continuation.finish(throwing: LiteRTLMError.inferenceFailure("No persistent session open — call openSession() first"))
                    return
                }

                let streamDone = DispatchSemaphore(value: 0)
                let state = StreamCallbackState(continuation: continuation, doneSemaphore: streamDone)
                let statePtr = Unmanaged.passRetained(state).toOpaque()

                let result = input.withCString { textPtr -> Int32 in
                    var inputData = InputData(
                        type: kInputText,
                        data: UnsafeRawPointer(textPtr),
                        size: strlen(textPtr)
                    )
                    return litert_lm_session_generate_content_stream(
                        session, &inputData, 1,
                        { callbackData, chunk, isFinal, errorMsg in
                            guard let cbData = callbackData else { return }
                            let st = Unmanaged<StreamCallbackState>.fromOpaque(cbData)
                                .takeUnretainedValue()

                            let errorMessage: String? = {
                                guard let errorMsg else { return nil }
                                let msg = String(cString: errorMsg)
                                return msg.isEmpty ? nil : msg
                            }()

                            if let chunk, errorMessage == nil {
                                let text = String(cString: chunk)
                                if !text.isEmpty { st.continuation.yield(text) }
                            }

                            if isFinal || errorMessage != nil {
                                if let error = errorMessage {
                                    st.continuation.finish(throwing: LiteRTLMError.inferenceFailure(error))
                                } else {
                                    st.continuation.finish()
                                }
                                let semaphore = st.doneSemaphore
                                Unmanaged<StreamCallbackState>.fromOpaque(cbData).release()
                                semaphore.signal()
                            }
                        },
                        statePtr
                    )
                }

                if result != 0 {
                    Unmanaged<StreamCallbackState>.fromOpaque(statePtr).release()
                    continuation.finish(throwing: LiteRTLMError.inferenceFailure("Failed to start stream"))
                    return
                }

                streamDone.wait()
                self.logSessionBenchmark(session)
            }
        }
    }

    // MARK: - Benchmark

    /// Retrieve benchmark info from a session.
    public struct BenchmarkInfo: Sendable {
        public let initTime: Double
        public let timeToFirstToken: Double
        public let prefillTurns: [(tokenCount: Int, tokensPerSec: Double)]
        public let decodeTurns: [(tokenCount: Int, tokensPerSec: Double)]
    }

    // MARK: - Private: Session-based Inference

    private func runSessionInference(
        prompt: String,
        temperature: Float,
        maxTokens: Int32
    ) async throws -> String {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<String, any Error>) in
            self.inferenceQueue.async { [self] in
                do {
                    guard let eng = self.engine else { throw LiteRTLMError.modelNotLoaded }

                    let (session, sessionConfig) = try self.createSession(
                        engine: eng, temperature: temperature, maxTokens: maxTokens
                    )
                    defer {
                        litert_lm_session_delete(session)
                        litert_lm_session_config_delete(sessionConfig)
                    }

                    let output = prompt.withCString { textPtr -> String? in
                        var input = InputData(
                            type: kInputText,
                            data: UnsafeRawPointer(textPtr),
                            size: strlen(textPtr)
                        )
                        guard let responses = litert_lm_session_generate_content(session, &input, 1) else {
                            return nil
                        }
                        defer { litert_lm_responses_delete(responses) }
                        return self.extractResponseText(responses)
                    }

                    guard let result = output else {
                        throw LiteRTLMError.inferenceFailure("generate_content returned no output")
                    }

                    self.logSessionBenchmark(session)
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private func runSessionInferenceStreaming(
        prompt: String,
        temperature: Float,
        maxTokens: Int32
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            self.inferenceQueue.async { [self] in
                do {
                    try self.ensureReady()
                    guard let eng = self.engine else {
                        continuation.finish(throwing: LiteRTLMError.modelNotLoaded)
                        return
                    }

                    let (session, sessionConfig) = try self.createSession(
                        engine: eng, temperature: temperature, maxTokens: maxTokens
                    )

                    let streamDone = DispatchSemaphore(value: 0)
                    let state = StreamCallbackState(continuation: continuation, doneSemaphore: streamDone)
                    let statePtr = Unmanaged.passRetained(state).toOpaque()

                    let result = prompt.withCString { textPtr -> Int32 in
                        var input = InputData(
                            type: kInputText,
                            data: UnsafeRawPointer(textPtr),
                            size: strlen(textPtr)
                        )
                        return litert_lm_session_generate_content_stream(
                            session, &input, 1,
                            { callbackData, chunk, isFinal, errorMsg in
                                guard let cbData = callbackData else { return }
                                let st = Unmanaged<StreamCallbackState>.fromOpaque(cbData)
                                    .takeUnretainedValue()

                                let errorMessage: String? = {
                                    guard let errorMsg else { return nil }
                                    let msg = String(cString: errorMsg)
                                    return msg.isEmpty ? nil : msg
                                }()

                                if let chunk, errorMessage == nil {
                                    let text = String(cString: chunk)
                                    if !text.isEmpty { st.continuation.yield(text) }
                                }

                                if isFinal || errorMessage != nil {
                                    if let error = errorMessage {
                                        st.continuation.finish(throwing: LiteRTLMError.inferenceFailure(error))
                                    } else {
                                        st.continuation.finish()
                                    }
                                    let semaphore = st.doneSemaphore
                                    Unmanaged<StreamCallbackState>.fromOpaque(cbData).release()
                                    semaphore.signal()
                                }
                            },
                            statePtr
                        )
                    }

                    if result != 0 {
                        Unmanaged<StreamCallbackState>.fromOpaque(statePtr).release()
                        litert_lm_session_delete(session)
                        litert_lm_session_config_delete(sessionConfig)
                        continuation.finish(throwing: LiteRTLMError.inferenceFailure("Failed to start stream"))
                        return
                    }

                    streamDone.wait()
                    self.logSessionBenchmark(session)
                    litert_lm_session_delete(session)
                    litert_lm_session_config_delete(sessionConfig)
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Private Helpers

    private func ensureReady() throws {
        guard status == .ready else { throw LiteRTLMError.modelNotLoaded }
    }

    private func createSession(
        engine eng: OpaquePointer,
        temperature: Float,
        maxTokens: Int32
    ) throws -> (session: OpaquePointer, config: OpaquePointer) {
        guard let sessionConfig = litert_lm_session_config_create() else {
            throw LiteRTLMError.inferenceFailure("Failed to create session config")
        }

        litert_lm_session_config_set_max_output_tokens(sessionConfig, maxTokens)
        var samplerParams = LiteRtLmSamplerParams(
            type: kTopP, top_k: 40, top_p: 0.95,
            temperature: temperature, seed: 0
        )
        litert_lm_session_config_set_sampler_params(sessionConfig, &samplerParams)

        guard let session = litert_lm_engine_create_session(eng, sessionConfig) else {
            litert_lm_session_config_delete(sessionConfig)
            throw LiteRTLMError.inferenceFailure("Failed to create session")
        }

        return (session, sessionConfig)
    }

    private func extractResponseText(_ responses: OpaquePointer) -> String? {
        let numCandidates = litert_lm_responses_get_num_candidates(responses)
        guard numCandidates > 0,
              let resultPtr = litert_lm_responses_get_response_text_at(responses, 0) else {
            return nil
        }
        return String(cString: resultPtr)
    }

    private func logSessionBenchmark(_ session: OpaquePointer) {
        guard let info = litert_lm_session_get_benchmark_info(session) else { return }
        defer { litert_lm_benchmark_info_delete(info) }

        let initTime = litert_lm_benchmark_info_get_total_init_time_in_second(info)
        let ttft = litert_lm_benchmark_info_get_time_to_first_token(info)
        let numDecode = litert_lm_benchmark_info_get_num_decode_turns(info)
        let numPrefill = litert_lm_benchmark_info_get_num_prefill_turns(info)

        Self.log.info("Benchmark: init=\(String(format: "%.2f", initTime))s, TTFT=\(String(format: "%.2f", ttft))s")

        for i in 0..<numPrefill {
            let tps = litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(info, Int32(i))
            let count = litert_lm_benchmark_info_get_prefill_token_count_at(info, Int32(i))
            Self.log.info("  Prefill[\(i)]: \(count) tokens @ \(String(format: "%.1f", tps)) tok/s")
        }
        for i in 0..<numDecode {
            let tps = litert_lm_benchmark_info_get_decode_tokens_per_sec_at(info, Int32(i))
            let count = litert_lm_benchmark_info_get_decode_token_count_at(info, Int32(i))
            Self.log.info("  Decode[\(i)]: \(count) tokens @ \(String(format: "%.1f", tps)) tok/s")
        }
    }

    // MARK: - Vision Helpers

    /// Convert any image format to JPEG and resize for vision inference.
    nonisolated static func prepareImageForVision(_ data: Data, maxDimension: Int = 1024) -> Data? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return nil }

        let width = cgImage.width
        let height = cgImage.height

        let maxDim = maxDimension
        let scale: Double
        if width > height {
            scale = width > maxDim ? Double(maxDim) / Double(width) : 1.0
        } else {
            scale = height > maxDim ? Double(maxDim) / Double(height) : 1.0
        }

        let targetWidth = Int(Double(width) * scale)
        let targetHeight = Int(Double(height) * scale)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil,
            width: targetWidth,
            height: targetHeight,
            bitsPerComponent: 8,
            bytesPerRow: targetWidth * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { return nil }

        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight))

        guard let resizedImage = context.makeImage() else { return nil }

        let mutableData = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            mutableData, "public.jpeg" as CFString, 1, nil
        ) else { return nil }

        let options: [CFString: Any] = [kCGImageDestinationLossyCompressionQuality: 0.85]
        CGImageDestinationAddImage(destination, resizedImage, options as CFDictionary)

        guard CGImageDestinationFinalize(destination) else { return nil }
        return mutableData as Data
    }

    nonisolated static func buildVisionMessageJSON(imagePath: String, text: String) -> String {
        let message: [String: Any] = [
            "role": "user",
            "content": [
                ["type": "image", "path": imagePath],
                ["type": "text", "text": text]
            ]
        ]
        guard let jsonData = try? JSONSerialization.data(withJSONObject: message),
              let jsonString = String(data: jsonData, encoding: .utf8) else {
            return #"{"role":"user","content":[{"type":"image","path":"\#(imagePath)"},{"type":"text","text":"\#(text)"}]}"#
        }
        return jsonString
    }

    nonisolated static func buildMultiImageMessageJSON(imagePaths: [String], text: String) -> String {
        var contentItems: [[String: Any]] = imagePaths.map { ["type": "image", "path": $0] }
        contentItems.append(["type": "text", "text": text])
        let message: [String: Any] = ["role": "user", "content": contentItems]
        guard let jsonData = try? JSONSerialization.data(withJSONObject: message),
              let jsonString = String(data: jsonData, encoding: .utf8) else {
            return buildVisionMessageJSON(imagePath: imagePaths.first ?? "", text: text)
        }
        return jsonString
    }

    nonisolated static func extractTextFromConversationResponse(_ json: String) -> String {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return json.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        if let content = obj["content"] as? [[String: Any]] {
            let texts = content.compactMap { $0["text"] as? String }
            if !texts.isEmpty { return texts.joined(separator: " ") }
        }

        if let candidates = obj["candidates"] as? [[String: Any]],
           let first = candidates.first,
           let content = first["content"] as? [String: Any],
           let parts = content["parts"] as? [[String: Any]] {
            let texts = parts.compactMap { $0["text"] as? String }
            if !texts.isEmpty { return texts.joined(separator: " ") }
        }

        if let text = obj["text"] as? String { return text }

        return json.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

// MARK: - Stream Callback State

private final class StreamCallbackState: @unchecked Sendable {
    let continuation: AsyncThrowingStream<String, Error>.Continuation
    let doneSemaphore: DispatchSemaphore

    init(continuation: AsyncThrowingStream<String, Error>.Continuation,
         doneSemaphore: DispatchSemaphore) {
        self.continuation = continuation
        self.doneSemaphore = doneSemaphore
    }
}

// MARK: - Errors

public enum LiteRTLMError: LocalizedError {
    case modelNotFound
    case modelNotLoaded
    case engineCreationFailed(String)
    case inferenceFailure(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound:
            "LiteRT-LM model file not found"
        case .modelNotLoaded:
            "LiteRT-LM model is not loaded — call load() first"
        case .engineCreationFailed(let detail):
            "Failed to create LiteRT-LM engine: \(detail)"
        case .inferenceFailure(let detail):
            "LiteRT-LM inference failed: \(detail)"
        }
    }
}
