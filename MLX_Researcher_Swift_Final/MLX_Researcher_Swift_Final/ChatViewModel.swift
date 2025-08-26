import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import SwiftUI
import Combine

struct ClassifyResponse: Decodable {
    let topic: String                // "on-topic" | "off-topic"
    let context_chunks: [String]?    // [] or nil when off-topic
}

@MainActor
class ChatViewModel: ObservableObject {
    @Published var input = ""
    @Published var messages: [String] = []
    @Published var isReady = true

    private var session: ChatSession?
    private let classifyURL = URL(string: "http://127.0.0.1:8000/classify")!

    // Your original system prompt (unchanged)
    private let SYSTEM_PROMPT = "You are an expert who only teaches data activism and Python programming to K–12 students. You explain concepts step by step using clear, scaffolded language. You never provide exact code solutions. If a student submits code with question marks (?), explain what each line is supposed to do by guiding them with detailed conceptual steps. For general programming questions (like \"How do I create a function?\"), give a full explanation with a short example, but do not solve specific problems. If a student asks something unrelated or off-topic, politely redirect them to focus on data activism or Python programming."
    
    init() {
        Task {
            // Do NOT pass instructions here — we’ll build the exact training template by hand
            let model = try await loadModel(id: "ShukraJaliya/BLUECOMPUTER.2")
            session = ChatSession(model, generateParameters: .init(maxTokens: 512, temperature: 0.65, topP: 0.9))
        }
    }

    func send() {
        guard let session = session else { return }
        let userText = input                 // EXACT input, no trimming or cleaning
        guard !userText.isEmpty else { return }

        messages.append("You: \(userText)")
        input = ""
        isReady = false

        Task { @MainActor in
            let start = Date()
            do {
                // 1) Classify (server prints on/off-topic in temp.py)
                var req = URLRequest(url: classifyURL)
                req.httpMethod = "POST"
                req.setValue("application/json", forHTTPHeaderField: "Content-Type")
                req.httpBody = try JSONEncoder().encode(["question": userText])

                let (data, resp) = try await URLSession.shared.data(for: req)
                guard let http = resp as? HTTPURLResponse, 200..<300 ~= http.statusCode else {
                    throw URLError(.badServerResponse)
                }
                let cls = try JSONDecoder().decode(ClassifyResponse.self, from: data)

                // 2) On-topic → include chunks; Off-topic → empty context
                let chunks = (cls.topic == "on-topic") ? (cls.context_chunks ?? []) : []
                let contextText = chunks.joined(separator: "\n")

                // 3) Build the prompt in the SAME template used during training
                //    (system → user → assistant, with <|im_start|> / <|im_end|>)
                let prompt: String
                if contextText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    // FREESTYLE: No context added for weak/empty retrieval
                    prompt = "<|im_start|>system \(SYSTEM_PROMPT) If you don’t have enough context, answer as best you can based on your training.<|im_end|><|im_start|>user \(userText)<|im_end|><|im_start|>assistant"
                } else {
                    // Normal: Use context as usual
                    prompt = "<|im_start|>system \(SYSTEM_PROMPT)<|im_end|><|im_start|>user \(userText)\n\nContext:\n\(contextText)<|im_end|><|im_start|>assistant"
                }

                // 4) Generate locally
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
