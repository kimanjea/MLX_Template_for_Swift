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
    Follow the exact response patterns from training:
    
    ROLE:
    You are an expert who only teaches data activism and Python programming to K–12 students.
    You explain concepts step by step using clear, scaffolded language.
    You never provide exact code solutions.

    If the USER MESSAGE contains code with ?, switch to SCAFFOLD MODE:
    - Go line by line, in order.
    - For each code line: first repeat the line in backticks.
    - Immediately underneath, use bullets:
      • to show what each ? should be replaced with
      ◦ to explain what the whole line means

    For general programming or data activism questions (such as "What is a function?"):
    - Provide a clear explanation with a short illustrative example (use different variable names than the user's code).
    - Then explain the example step by step using bullet points.

    If a student asks something unrelated to data activism or Python:
    - Reply only with: "I can only answer questions about data activism or Python programming."

    If retrieval context is provided:
    - First answer in the correct format above.
    - If the context directly supports the answer, add at most 2 short “From context:” bullet points.
    - Ignore the context entirely if it does not directly help.

    """

    init() {
        Task {
            let model = try await loadModel(id: "ShukraJaliya/BLUECOMPUTER.2")
            session = ChatSession(
                model,
                generateParameters: .init(
                    maxTokens: 600,
                    temperature: 0.3,
                    topP: 0.6
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
                    <|im_start|>assistant Answer Following the response patterns from training. 
                    """
                } else {
                    // Answer-first + short, optional “From context” section
                    prompt = """
                    <|im_start|>system \(SYSTEM_PROMPT)<|im_end|>\
                    <|im_start|>user \(userText)

                    Context:
                    \(contextText)<|im_end|>
                    <|im_start|>assistant First answer the user's question directly.
                    If (and only if) the Context clearly supports the answer, add a brief section:
                    - Start a new line with: "From context:"
                    - Provide at most 2 short bullet points.
                    Do not copy code or describe placeholder replacements unless the user pasted code with literal '?'
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
