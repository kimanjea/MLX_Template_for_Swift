import SwiftUI

struct ContentView: View {
    @StateObject private var vm = ChatViewModel()

    // Sidebar suggestions
    let sidebarSuggestions = [
        "What is data activism?",
        "Who are important figures in data activism?",
        "How can I become a data activist?",
        "What is black box programming?",
        "What are some basic coding concepts?"
    ]
    
    // Bottom quick actions
    let bottomSuggestions = [
        ["Lesson recall", "Summarize concepts", "Provide an example", "Quiz Me"],
        ["Current events", "Data Activism Quotes", "Important figures"]
    ]

    var body: some View {
        ZStack {
            // Background layer
            Image("Background")
                .resizable()
                .aspectRatio(contentMode: .fill)
                .ignoresSafeArea()

            HStack(spacing: 0) {
                // Sidebar
                VStack(alignment: .leading, spacing: 0) {
                    // Header with logo and title
                    HStack(spacing: 12) {
                        Image("Logo")
                            .resizable()
                            .frame(width: 44, height: 44)
                            .cornerRadius(10)
                        Text("AVELA AI")
                            .font(.largeTitle.bold())
                            .foregroundColor(.black)
                        Spacer()
                    }
                    .padding(.horizontal, 20)
                    .padding(.vertical, 20)

                    Rectangle()
                        .fill(Color.black.opacity(0.2))
                        .frame(height: 1)
                        .padding(.horizontal, 20)

                    Text("Conversations")
                        .font(.title2.weight(.medium))
                        .foregroundColor(.black)
                        .padding(.horizontal, 20)
                        .padding(.vertical, 16)

                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 12) {
                            ForEach(sidebarSuggestions, id: \.self) { suggestion in
                                Button(action: {
                                    vm.input = suggestion
                                }) {
                                    Text(suggestion)
                                        .font(.system(size: 14, weight: .medium))
                                        .padding(.vertical, 12)
                                        .padding(.horizontal, 16)
                                        .frame(maxWidth: .infinity)
                                        .background(
                                            Capsule()
                                                .stroke(Color.black.opacity(0.3), lineWidth: 1.5)
                                        )
                                        .foregroundColor(.black)
                                        .multilineTextAlignment(.center)
                                }
                            }
                        }
                        .padding(.vertical)
                    }
                }
                .frame(width: 320)

                // Separator
                Rectangle()
                    .fill(Color.black.opacity(0.2))
                    .frame(width: 1)

                // Main chat area
                VStack(spacing: 0) {
                    // Top bar with settings icon
                    HStack {
                        Spacer()
                        Button(action: {}) {
                            Image(systemName: "gearshape")
                                .font(.title2)
                                .foregroundColor(.gray)
                        }
                        .padding(.trailing, 30)
                        .padding(.top, 30)
                    }

                    // Chat history
                    ScrollViewReader { proxy in
                        ScrollView {
                            VStack(alignment: .leading, spacing: 12) {
                                ForEach(vm.messages.indices, id: \.self) { idx in
                                    let msg = vm.messages[idx]
                                    HStack {
                                        if msg.starts(with: "You:") { Spacer() }

                                        Text(msg
                                            .replacingOccurrences(of: "You: ", with: "")
                                            .replacingOccurrences(of: "Bot: ", with: "")
                                        )
                                        .padding(10)
                                        .background(
                                            RoundedRectangle(cornerRadius: 8)
                                                .fill(msg.starts(with: "You:") ?
                                                      Color.blue.opacity(0.1) :
                                                      Color.gray.opacity(0.1))
                                        )
                                        .frame(maxWidth: .infinity,
                                               alignment: msg.starts(with: "You:") ? .trailing : .leading)

                                        if !msg.starts(with: "You:") { Spacer() }
                                    }
                                    .id(idx)
                                }
                            }
                            .padding()
                        }
                        .onChange(of: vm.messages.count) { _ in
                            if let last = vm.messages.indices.last {
                                withAnimation {
                                    proxy.scrollTo(last, anchor: .bottom)
                                }
                            }
                        }
                    }

                    // Input field and send button
                    HStack {
                        TextField("Type a message…", text: $vm.input)
                            .textFieldStyle(.roundedBorder)
                        Button("Send") {
                            vm.send()
                        }
                        .disabled(vm.input.trimmingCharacters(in: .whitespaces).isEmpty)
                    }
                    .padding()

                    // Bottom suggestions
                    VStack(spacing: 12) {
                        ForEach(bottomSuggestions, id: \.self) { row in
                            HStack(spacing: 12) {
                                ForEach(row, id: \.self) { suggestion in
                                    Button(action: {
                                        vm.input = suggestion
                                    }) {
                                        Text(suggestion)
                                            .font(.system(size: 14, weight: .medium))
                                            .padding(.vertical, 12)
                                            .padding(.horizontal, 16)
                                            .frame(maxWidth: .infinity)
                                            .background(
                                                Capsule()
                                                    .stroke(Color.black.opacity(0.3), lineWidth: 1.5)
                                            )
                                            .foregroundColor(.black)
                                    }
                                }
                            }
                        }
                    }
                    .padding(.horizontal, 30)
                    .padding(.bottom, 12)
                }
                .frame(maxWidth: .infinity)
            }
            .background(
                RoundedRectangle(cornerRadius: 30)
                    .fill(Color(red: 0xFC/255, green: 0xFC/255, blue: 0xF9/255))
                    .stroke(Color.black, lineWidth: 2)
            )
            .padding(.horizontal, 40)
            .padding(.vertical, 30)
        }
    }
}

@main
struct MLX_templateApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
