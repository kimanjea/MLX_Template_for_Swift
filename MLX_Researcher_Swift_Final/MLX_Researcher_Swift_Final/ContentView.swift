import SwiftUI

struct ContentView: View {
    @StateObject private var vm = ChatViewModel()

    // Updated suggestions to match reference
    let sidebarSuggestions = [
        "What is data activism?",
        "Who are important figures in data activism?",
        "How can I become a data activist?",
        "What is black box programming?",
        "What are some basic coding concepts?"
    ]
    
    let bottomSuggestions = [
        ["Lesson recall", "Summarize concepts", "Provide an example", "Quiz Me"],
        ["Current events", "Data Activism Quotes", "Important figures"]
    ]

    var body: some View {
        ZStack {
            // Background image at the bottom layer
            Image("Background")
                .resizable()
                .aspectRatio(contentMode: .fill)
                .ignoresSafeArea(.all)
            
            // Single container for everything
            HStack(spacing: 0) {
                // Sidebar section
                VStack(alignment: .leading, spacing: 0) {
                    // Header section
                    VStack(alignment: .leading, spacing: 16) {
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
                    }
                    .padding(.horizontal, 20)
                    .padding(.vertical, 20)
                    
                    // Separator line
                    Rectangle()
                        .fill(Color.black.opacity(0.2))
                        .frame(height: 1)
                        .padding(.horizontal, 20)
                    
                    // Conversations section
                    VStack(alignment: .leading, spacing: 16) {
                        Text("Conversations")
                            .font(.title2.weight(.medium))
                            .foregroundColor(.black)
                    }
                    .padding(.horizontal, 20)
                    .padding(.vertical, 20)
                    
                    // Scrollable suggestions
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
                                        .frame(maxWidth: .infinity, alignment: .center)
                                        .background(
                                            Capsule()
                                                .stroke(Color.black.opacity(0.3), lineWidth: 1.5)
                                                .background(Capsule().fill(Color.clear))
                                        )
                                        .foregroundColor(.black)
                                        .multilineTextAlignment(.center)
                                }
                                .buttonStyle(PlainButtonStyle())
                            }
                        }
                        .padding(.horizontal, 20)
                        .padding(.vertical, 8)
                    }

                    Spacer()
                }
                .frame(width: 320)
                
                // Vertical separator between sidebar and main content
                Rectangle()
                    .fill(Color.black.opacity(0.2))
                    .frame(width: 1)
                
                // Main content area
                VStack {
                    Spacer()
                    VStack(spacing: 30) {
                        HStack {
                            Spacer()
                            Button(action: {}) {
                                Image(systemName: "gearshape")
                            }
                            .font(.title2)
                            .foregroundColor(.gray)
                        }
                        .padding([.top, .trailing], 30)

                        VStack(spacing: 16) {
                            Text("Welcome Back!")
                                .font(.system(size: 48, weight: .bold))
                                .foregroundColor(.black)
                                .frame(maxWidth: .infinity, alignment: .center)

                            Text("Ready to learn more about data activism?")
                                .font(.title2)
                                .foregroundColor(.black.opacity(0.7))
                                .frame(maxWidth: .infinity, alignment: .center)
                        }
                        .padding(.horizontal, 30)

                        VStack(spacing: 20) {
                            ZStack {
                                RoundedRectangle(cornerRadius: 25)
                                    .fill(Color.white)
                                    .shadow(color: Color.black.opacity(0.1), radius: 8, x: 0, y: 4)
                                    .frame(height: 560)
                                
                                HStack {
                                    ZStack(alignment: .topLeading) {
                                        // TextEditor
                                        TextEditor(text: $vm.input)
                                            .font(.system(size: 16))
                                            .foregroundColor(.black)
                                            .background(Color.clear)
                                            .scrollContentBackground(.hidden)
                                            .padding(.horizontal, 4)
                                            .padding(.vertical, 8)
                                        
                                        // Placeholder text inside the white rectangle
                                        if vm.input.isEmpty {
                                            Text("Insert Text...")
                                                .foregroundColor(.gray.opacity(0.6))
                                                .font(.system(size: 30))
                                                .padding(.horizontal, 4)
                                                .padding(.vertical, 8)
                                                .allowsHitTesting(false)
                                        }
                                    }
                                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                                    
                                    VStack {
                                        Spacer()
                                        Button("Submit") {
                                            vm.send()
                                        }
                                        .font(.system(size: 16, weight: .semibold))
                                        .foregroundColor(.white)
                                        .padding(.horizontal, 28)
                                        .padding(.vertical, 12)
                                        .background(
                                            Capsule()
                                                .fill(Color.orange)
                                        )
                                        .disabled(vm.input.trimmingCharacters(in: .whitespaces).isEmpty)
                                    }
                                    .padding(.bottom, 24)
                                }
                                .padding(.horizontal, 20)
                                .padding(.vertical, 16)
                            }
                            .padding(.horizontal, 30)
                            
                            // Suggestion buttons
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
                                                            .background(Capsule().fill(Color.clear))
                                                    )
                                                    .foregroundColor(.black)
                                            }
                                            .buttonStyle(PlainButtonStyle())
                                        }
                                    }
                                }
                            }
                            .padding(.horizontal, 30)
                            .padding(.bottom, 12) // Changed padding from 30 to 12 to move bubbles closer
                        }
                    }
                    
                    Spacer()
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

extension Array {
    func chunked(into size: Int) -> [[Element]] {
        var chunks: [[Element]] = []
        var index = 0
        while index < self.count {
            let end = Swift.min(index + size, self.count)
            chunks.append(Array(self[index..<end]))
            index += size
        }
        return chunks
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
