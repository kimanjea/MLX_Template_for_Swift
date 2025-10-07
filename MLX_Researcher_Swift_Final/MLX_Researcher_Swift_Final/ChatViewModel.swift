import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import SwiftUI
import Combine

struct ClassifyResponse: Decodable {
    let topic: String                 // "on-topic" | "off-topic"
    let context_chunks: [String]?     // [] or nil when off-topic
}

@MainActor
class ChatViewModel: ObservableObject {
    @Published var input = ""
    @Published var messages: [String] = []
    @Published var isReady = true

    private var session: ChatSession?
    private let classifyURL = URL(string: "http://127.0.0.1:8000/classify")!

    // System prompt + guardrails about scaffold mode & context use
    private let SYSTEM_PROMPT = """
    You are an expert who only teaches data activism and Python programming to K–12 students. 
    You explain concepts step by step using clear, scaffolded language. 
    You never provide exact code solutions. 
    If a student submits code with question marks (?), explain what each line is supposed to do by guiding them with detailed conceptual steps. 
    For general programming questions (like "How to create a function?"), give a full explanation with a short example, but do not solve specific problems. 
    If a student asks something unrelated or off-topic, politely redirect them to focus on data activism or Python programming.
    """

    init() {
        Task {
            let model = try await loadModel(id: "ShukraJaliya/BLUECOMPUTER.2")
            session = ChatSession(
                model,
                generateParameters: .init(
                    maxTokens: 600,
                    temperature: 0.4,
                    topP: 0.8
                )
            )
        }
    }

    func send() {
        guard let session = session else { return }
        let userText = input                 // EXACT input, no trimming/cleaning
        guard !userText.isEmpty else { return }

        messages.append("You: \(userText)")
        input = ""
        isReady = false

        Task { @MainActor in
            let start = Date()
            do {
                // 1) Classify
                var req = URLRequest(url: classifyURL)
                req.httpMethod = "POST"
                req.setValue("application/json", forHTTPHeaderField: "Content-Type")
                req.httpBody = try JSONEncoder().encode(["question": userText])

                let (data, resp) = try await URLSession.shared.data(for: req)
                guard let http = resp as? HTTPURLResponse, 200..<300 ~= http.statusCode else {
                    throw URLError(.badServerResponse)
                }
                let cls = try JSONDecoder().decode(ClassifyResponse.self, from: data)
                let chunks = (cls.topic == "on-topic") ? (cls.context_chunks ?? []) : []
                var contextText = chunks.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)

                // If the question looks like a code chunk with '?', force contextText empty (no RAG for code explanation)
                let isCodeQuestionWithUnknowns = userText.contains("?") && userText.contains(":")
                if isCodeQuestionWithUnknowns {
                    contextText = ""
                }

                print("[Swift RAG debug] topic=\(cls.topic)  context.len=\(contextText.count)")

                // 2) Build training-aligned prompt (<|im_start|>/end)
                let prompt: String
                if contextText.isEmpty {
                    // Conceptual answer, no RAG
                    prompt = """
                    <|im_start|>system \(SYSTEM_PROMPT)<|im_end|>\
                    <|im_start|>user \(userText)<|im_end|>
                    <|im_start|>assistant
                    """
                } else {
                    // Answer-first + short, optional “From context” section
                    prompt = """
                    <|im_start|>system \(SYSTEM_PROMPT)If the provided context is directly relevant, smoothly weave up to two supporting details from it into your explanation. Do not highlight or label these as 'context'—just incorporate them naturally. Do not copy code or describe placeholder replacements unless the user pasted code with literal '?'.<|im_end|>\
                    <|im_start|>user \(userText)

                    Context:
                    \(contextText)<|im_end|>
                    <|im_start|>assistant
                    """
                }

                print("[Prompt sent to model]:\n\(prompt)")

                // 3) Generate locally
                let reply = try await session.respond(to: prompt)

                let elapsed = Date().timeIntervalSince(start)
                messages.append("Bot (\(String(format: "%.2f", elapsed))s): \(reply)")
            } catch {
                let elapsed = Date().timeIntervalSince(start)
                messages.append("Error (\(String(format: "%.2f", elapsed))s): \(error.localizedDescription)")
            }
            isReady = true
        }
    }
}
