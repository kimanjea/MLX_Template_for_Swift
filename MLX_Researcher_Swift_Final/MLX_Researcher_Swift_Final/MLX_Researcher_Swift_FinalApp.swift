//
//  MLX_Researcher_Swift_FinalApp.swift
//  MLX_Researcher_Swift_Final
//
//  Created by xrlead on 8/5/25.
//

import SwiftUI
import SwiftData

@main
struct MLX_Researcher_Swift_FinalApp: App {
    var sharedModelContainer: ModelContainer = {
        let schema = Schema([
            Item.self,
        ])
        let modelConfiguration = ModelConfiguration(schema: schema, isStoredInMemoryOnly: false)

        do {
            return try ModelContainer(for: schema, configurations: [modelConfiguration])
        } catch {
            fatalError("Could not create ModelContainer: \(error)")
        }
    }()

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .modelContainer(sharedModelContainer)
    }
}
