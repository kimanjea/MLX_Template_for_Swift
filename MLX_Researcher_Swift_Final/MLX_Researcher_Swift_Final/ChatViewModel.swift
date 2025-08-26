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

    Follow the exact response patterns from training:

    1. If the USER MESSAGE contains code with literal '?' placeholders:
       - Repeat the code line.
       - On the next line, use bullet points (• and ◦) to explain each placeholder and the meaning of the line.
       - Do this line by line, never grouping explanations together.

    2. If the USER MESSAGE is a general programming or data activism question:
       - Give a short illustrative example (with different variable names than the user’s code).
       - Then explain the example step by step using bullet points.

    3. If the USER MESSAGE is unrelated to data activism or Python:
       - Reply only with: "I can only answer questions about data activism or Python programming."

    4. If retrieval context is provided:
       - First answer in the correct format above.
       - If the context directly supports the answer, add at most 2 short “From context:” bullet points.
       - Ignore the context if it is not directly helpful.
    """

    init() {
        Task {
            let model = try await loadModel(id: "ShukraJaliya/BLUECOMPUTER.2")
            session = ChatSession(
                model,
                generateParameters: .init(
                    maxTokens: 600,
                    temperature: 0.7,
                    topP: 0.9
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
                let contextText = chunks.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)

                print("[Swift RAG debug] topic=\(cls.topic)  context.len=\(contextText.count)")

                // 2) Build training-aligned prompt (<|im_start|>/end)
                let prompt: String
                if contextText.isEmpty {
                    // Conceptual answer, no RAG
                    prompt = """
                    <|im_start|>system \(SYSTEM_PROMPT)<|im_end|>\
                    <|im_start|>user \(userText)<|im_end|>
                    <|im_start|>assistant Please answer the user's question directly in 2–4 sentences. Do not invent code or placeholder replacements.
                    """
                } else {
                    // Answer-first + short, optional “From context” section
                    prompt = """
                    <|im_start|>system \(SYSTEM_PROMPT)<|im_end|>\
                    <|im_start|>user \(userText)

                    Context:
                    \(contextText)<|im_end|>
                    <|im_start|>assistant First answer the user's question directly in 2–4 sentences.
                    If (and only if) the Context clearly supports the answer, add a brief section:
                    - Start a new line with: "From context:"
                    - Provide at most 2 short bullet points.
                    Do not copy code or describe placeholder replacements unless the user pasted code with literal '?' placeholders.
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
