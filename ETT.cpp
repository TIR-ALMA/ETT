#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <csignal>
#include <sstream>
#include <algorithm>
#include <memory>
#include <array>

namespace fs = std::filesystem;

class NeuralNetworkController {
private:
    std::string pythonScriptPath;
    std::string dataPath;
    int epochs;
    bool isTraining;
    int trainingPid;
    std::unique_ptr<std::thread> monitorThread;
    volatile bool shouldStopMonitor;

    // Function to read output from a command
    std::string exec(const std::string& cmd) {
        std::array<char, 128> buffer;
        std::string result;
        std::unique_ptr<FILE, int(*)(FILE*)> pipe(popen(cmd.c_str(), "r"), pclose);
        if (!pipe) {
            throw std::runtime_error("popen() failed!");
        }
        while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
            result += buffer.data();
        }
        // Remove trailing newline
        if (!result.empty() && result.back() == '\n') {
            result.pop_back();
        }
        return result;
    }

    // Monitor thread function to track child process
    void monitorTrainingProcess() {
        while (!shouldStopMonitor) {
            if (trainingPid > 0) {
                std::string checkCmd = "ps -p " + std::to_string(trainingPid) + " >/dev/null 2>&1 && echo 1 || echo 0";
                try {
                    std::string result = exec(checkCmd);
                    if (result != "1") {
                        // Process has ended
                        isTraining = false;
                        trainingPid = -1;
                        break;
                    }
                } catch (...) {
                    // If we can't check, assume it's still running
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

public:
    NeuralNetworkController() : isTraining(false), epochs(10), trainingPid(-1), shouldStopMonitor(false) {}

    ~NeuralNetworkController() {
        if (isTraining && trainingPid > 0) {
            stopTraining();
        }
        if (monitorThread && monitorThread->joinable()) {
            shouldStopMonitor = true;
            monitorThread->join();
        }
    }

    void runPythonScript(const std::string& script, const std::string& args) {
        std::string command = "python3 \"" + script + "\" " + args + " 2>&1 & echo $!";
        FILE* pipe = popen(command.c_str(), "r");
        if (pipe) {
            char buffer[16];
            if (fgets(buffer, sizeof(buffer), pipe)) {
                // Remove newlines and carriage returns
                std::string result(buffer);
                result.erase(std::remove(result.begin(), result.end(), '\n'), result.end());
                result.erase(std::remove(result.begin(), result.end(), '\r'), result.end());
                
                try {
                    trainingPid = std::stoi(result);
                } catch (...) {
                    trainingPid = -1;
                }
            }
            pclose(pipe);
        } else {
            trainingPid = -1;
        }
    }

    void startTraining() {
        if (isTraining) {
            std::cout << "Training is already running!" << std::endl;
            return;
        }

        if (dataPath.empty()) {
            std::cout << "Error: Data path not set. Use 'set_data' command first." << std::endl;
            return;
        }

        if (pythonScriptPath.empty()) {
            std::cout << "Error: Python script path not set." << std::endl;
            return;
        }

        if (!fs::exists(pythonScriptPath)) {
            std::cout << "Error: Python script does not exist: " << pythonScriptPath << std::endl;
            return;
        }

        isTraining = true;
        std::string args = "\"" + dataPath + "\" --epochs " + std::to_string(epochs);
        runPythonScript(pythonScriptPath, args);
        
        if (trainingPid > 0) {
            std::cout << "Started training with PID: " << trainingPid << std::endl;
            
            // Start monitoring thread
            shouldStopMonitor = false;
            monitorThread = std::make_unique<std::thread>(&NeuralNetworkController::monitorTrainingProcess, this);
        } else {
            std::cout << "Failed to start training process." << std::endl;
            isTraining = false;
            trainingPid = -1;
        }
    }

    void stopTraining() {
        if (!isTraining || trainingPid <= 0) {
            std::cout << "Training is not running!" << std::endl;
            return;
        }

        std::string killCmd = "kill " + std::to_string(trainingPid) + " 2>/dev/null";
        int result = system(killCmd.c_str());
        
        // Wait a bit for graceful termination
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Check if process still exists
        std::string checkCmd = "ps -p " + std::to_string(trainingPid) + " >/dev/null 2>&1 && echo 1 || echo 0";
        try {
            std::string result_check = exec(checkCmd);
            if (result_check == "1") {
                // Process still alive, try SIGKILL
                std::string forceKillCmd = "kill -9 " + std::to_string(trainingPid) + " 2>/dev/null";
                system(forceKillCmd.c_str());
                std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Wait for termination
            }
        } catch (...) {
            // If we can't check, assume it's terminated
        }
        
        isTraining = false;
        trainingPid = -1;
        
        // Stop monitor thread if running
        if (monitorThread && monitorThread->joinable()) {
            shouldStopMonitor = true;
            monitorThread->join();
        }
        
        std::cout << "Training stopped." << std::endl;
    }

    void chatWithModel() {
        if (pythonScriptPath.empty()) {
            std::cout << "Error: Python script path not set." << std::endl;
            return;
        }

        if (!fs::exists(pythonScriptPath)) {
            std::cout << "Error: Python script does not exist: " << pythonScriptPath << std::endl;
            return;
        }

        std::cout << "Starting chat with the model..." << std::endl;
        std::string chatArgs = "--chat";
        
        // For chat, we don't need to track the PID as we did for training
        std::string command = "python3 \"" + pythonScriptPath + "\" " + chatArgs;
        int result = system(command.c_str());
        
        if (result == 0) {
            std::cout << "Chat finished." << std::endl;
        } else {
            std::cout << "Failed to run chat process." << std::endl;
        }
    }

    void setPythonScript(const std::string& path) {
        pythonScriptPath = path;
    }

    void setDataPath(const std::string& path) {
        dataPath = path;
    }

    void setEpochs(int ep) {
        if (ep > 0) {
            epochs = ep;
        } else {
            std::cout << "Error: Epochs must be a positive number." << std::endl;
        }
    }

    bool getIsTraining() const {
        return isTraining;
    }

    int getTrainingPid() const {
        return trainingPid;
    }
};

bool fileExists(const std::string& filename) {
    return fs::exists(filename);
}

void createEmptyPyFile(const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << "# Auto-generated Python script for neural network\n";
        file << "import sys\n";
        file << "import argparse\n";
        file << "import time\n\n";
        file << "def main():\n";
        file << "    parser = argparse.ArgumentParser(description='Neural Network Script')\n";
        file << "    parser.add_argument('data_path', help='Path to training data')\n";
        file << "    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')\n";
        file << "    parser.add_argument('--chat', action='store_true', help='Start chat mode')\n";
        file << "    args = parser.parse_args()\n\n";
        file << "    if args.chat:\n";
        file << "        print('Chat mode activated...')\n";
        file << "        print('Enter messages (type \"quit\" to exit):')\n";
        file << "        while True:\n";
        file << "            user_input = input('> ')\n";
        file << "            if user_input.lower() in ['quit', 'exit']:\n";
        file << "                break\n";
        file << "            print(f'Model response: You said \"{user_input}\"')\n";
        file << "    else:\n";
        file << "        print(f'Starting training on {args.data_path} for {args.epochs} epochs')\n";
        file << "        for epoch in range(args.epochs):\n";
        file << "            print(f'Epoch {epoch+1}/{args.epochs}')\n";
        file << "            time.sleep(1)  # Simulate work\n";
        file << "        print('Training completed!')\n\n";
        file << "if __name__ == '__main__':\n";
        file << "    main()\n";
        file.close();
    }
}

std::vector<std::string> splitCommand(const std::string& input) {
    std::vector<std::string> tokens;
    std::istringstream iss(input);
    std::string token;
    
    // Properly handle quoted strings
    size_t pos = 0;
    while (pos < input.length()) {
        // Skip leading spaces
        while (pos < input.length() && input[pos] == ' ') pos++;
        if (pos >= input.length()) break;
        
        if (input[pos] == '"') {
            // Handle quoted string
            pos++; // skip opening quote
            size_t start = pos;
            while (pos < input.length() && input[pos] != '"') pos++;
            if (pos < input.length()) { // found closing quote
                tokens.push_back(input.substr(start, pos - start));
                pos++; // skip closing quote
            } else {
                // Unmatched quote - treat as regular string
                pos = start; // reset to after quote
                start = pos;
                while (pos < input.length() && input[pos] != ' ') pos++;
                tokens.push_back(input.substr(start, pos - start));
            }
        } else {
            // Handle unquoted string
            size_t start = pos;
            while (pos < input.length() && input[pos] != ' ') pos++;
            tokens.push_back(input.substr(start, pos - start));
        }
    }
    
    return tokens;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "ETT (Easy To Teach) - Neural Network Trainer" << std::endl;
        std::cout << "Usage: ett <script.py>" << std::endl;
        return 1;
    }

    std::string scriptName = argv[1];
    
    if (!fileExists(scriptName)) {
        createEmptyPyFile(scriptName);
        std::cout << "Created new file: " << scriptName << " (with template content)" << std::endl;
    } else {
        std::cout << "Using existing file: " << scriptName << std::endl;
    }

    NeuralNetworkController controller;
    controller.setPythonScript(scriptName);
    
    std::string input;
    std::cout << "\nETT initialized with: " << scriptName << std::endl;

    while (true) {
        std::cout << "\nCommands:" << std::endl;
        std::cout << "1. set_data [path] - Set data folder/file path" << std::endl;
        std::cout << "2. set_epochs [number] - Set number of epochs" << std::endl;
        std::cout << "3. start - Start training" << std::endl;
        std::cout << "4. stop - Stop training" << std::endl;
        std::cout << "5. chat - Chat with the model" << std::endl;
        std::cout << "6. status - Show training status" << std::endl;
        std::cout << "7. exit - Exit program" << std::endl;
        std::cout << "\nEnter command: ";

        std::getline(std::cin, input);

        // Handle commands with parameters
        if (input.substr(0, 8) == "set_data") {
            std::string path = input.substr(9);
            // Remove leading/trailing whitespace
            size_t start = path.find_first_not_of(" \t");
            size_t end = path.find_last_not_of(" \t");
            if (start != std::string::npos && end != std::string::npos) {
                path = path.substr(start, end - start + 1);
            } else if (start == std::string::npos) {
                path = "";
            }
            
            if (!path.empty()) {
                controller.setDataPath(path);
                std::cout << "Data path set to: " << path << std::endl;
            } else {
                std::cout << "No data path provided." << std::endl;
            }
        } else if (input.substr(0, 10) == "set_epochs") {
            std::string epochStr = input.substr(11);
            // Remove leading/trailing whitespace
            size_t start = epochStr.find_first_not_of(" \t");
            size_t end = epochStr.find_last_not_of(" \t");
            if (start != std::string::npos && end != std::string::npos) {
                epochStr = epochStr.substr(start, end - start + 1);
            } else if (start == std::string::npos) {
                epochStr = "";
            }
            
            if (!epochStr.empty()) {
                try {
                    int epochs = std::stoi(epochStr);
                    controller.setEpochs(epochs);
                    std::cout << "Epochs set to: " << epochs << std::endl;
                } catch (...) {
                    std::cout << "Invalid epoch number" << std::endl;
                }
            } else {
                std::cout << "No epoch number provided." << std::endl;
            }
        } else if (input == "start") {
            controller.startTraining();
        } else if (input == "stop") {
            controller.stopTraining();
        } else if (input == "chat") {
            controller.chatWithModel();
        } else if (input == "status") {
            std::cout << "Training status: " << (controller.getIsTraining() ? "Running" : "Stopped") << std::endl;
            if (controller.getIsTraining()) {
                std::cout << "Training PID: " << controller.getTrainingPid() << std::endl;
            }
        } else if (input == "exit") {
            if (controller.getIsTraining()) {
                std::cout << "Stopping training before exit..." << std::endl;
                controller.stopTraining();
            }
            break;
        } else {
            std::cout << "Unknown command" << std::endl;
        }
    }

    return 0;
}
