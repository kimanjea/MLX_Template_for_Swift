import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import SwiftUI
import Combine

struct ClassifyResponse: Decodable {
    let topic: String
    let context_chunks: [String]?
}

@MainActor
class ChatViewModel: ObservableObject {
    @Published var input = ""
    @Published var messages: [String] = []
    @Published var isReady = true

    var session: ChatSession?
    let classifyURL = URL(string: "http://127.0.0.1:8000/classify")!

    let SYSTEM_PROMPT = """
    You are an expert who only teaches data activism and Python programming to K–12 students.
    You explain concepts step by step using clear, scaffolded language.
    You never provide exact code solutions.
    If a student submits code with question marks (?), explain what each line is supposed to do by guiding them with detailed conceptual steps.
    For general programming questions (like "How do I create a function?"), give a detailed explanation with a short example, but do not solve specific student problems.
    """

    init() {
        Task {
            let model = try await loadModel(id: "ShukraJaliya/BLUECOMPUTER.2")
            session = ChatSession(model)
        }
    }

    func send() {
        guard let session = session else { return }
        let userText = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !userText.isEmpty else { return }

        messages.append("You: \(userText)")
        input = ""
        isReady = false

        Task { @MainActor in
            let start = Date()
            do {
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
                let contextText = chunks.joined(separator: "\n")

                let prompt = """
                <|im_start|>system
                \(SYSTEM_PROMPT)
                <|im_end|>
                <|im_start|>user
                Question:
                \(userText)

                Context:
                \(contextText)
                <|im_end|>
                <|im_start|>assistant
                """

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
