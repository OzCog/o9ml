//
// Storage Backend Test - Simulates atomspace-rocks functionality
//
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <string>

class StorageBackend {
private:
    std::string storage_path;
    std::map<std::string, std::string> key_value_store;
    
public:
    StorageBackend(const std::string& path) : storage_path(path) {}
    
    bool store(const std::string& key, const std::string& value) {
        key_value_store[key] = value;
        
        // Simulate persistent storage
        std::ofstream file(storage_path + "/" + key + ".dat");
        if (file.is_open()) {
            file << value;
            file.close();
            return true;
        }
        return false;
    }
    
    std::string retrieve(const std::string& key) {
        auto it = key_value_store.find(key);
        if (it != key_value_store.end()) {
            return it->second;
        }
        
        // Try loading from file
        std::ifstream file(storage_path + "/" + key + ".dat");
        if (file.is_open()) {
            std::string value((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
            key_value_store[key] = value;
            return value;
        }
        
        return "";
    }
    
    bool remove(const std::string& key) {
        key_value_store.erase(key);
        std::string filepath = storage_path + "/" + key + ".dat";
        return std::remove(filepath.c_str()) == 0;
    }
    
    std::vector<std::string> list_keys() {
        std::vector<std::string> keys;
        for (const auto& pair : key_value_store) {
            keys.push_back(pair.first);
        }
        return keys;
    }
    
    size_t size() const {
        return key_value_store.size();
    }
    
    bool validate_storage() {
        std::cout << "  Validating storage backend..." << std::endl;
        
        // Test basic operations
        std::string test_key = "test_atom_123";
        std::string test_value = R"({"type": "ConceptNode", "name": "TestConcept", "tv": {"strength": 0.8, "confidence": 0.7}})";
        
        if (!store(test_key, test_value)) {
            std::cerr << "    ERROR: Failed to store test data" << std::endl;
            return false;
        }
        
        std::string retrieved = retrieve(test_key);
        if (retrieved != test_value) {
            std::cerr << "    ERROR: Retrieved data doesn't match stored data" << std::endl;
            return false;
        }
        
        if (!remove(test_key)) {
            std::cerr << "    ERROR: Failed to remove test data" << std::endl;
            return false;
        }
        
        std::cout << "    ✓ Storage operations validated" << std::endl;
        std::cout << "    ✓ Current storage size: " << size() << " entries" << std::endl;
        return true;
    }
};

int main() {
    std::cout << "Testing storage backend functionality..." << std::endl;
    
    // Create temporary storage directory
    system("mkdir -p /tmp/atomspace_storage");
    
    StorageBackend storage("/tmp/atomspace_storage");
    
    // Store hypergraph data
    storage.store("atom_1", R"({"type": "ConceptNode", "name": "Cat", "tv": {"strength": 0.95, "confidence": 0.9}})");
    storage.store("atom_2", R"({"type": "ConceptNode", "name": "Animal", "tv": {"strength": 0.9, "confidence": 0.8}})");
    storage.store("link_1", R"({"type": "InheritanceLink", "outgoing": ["atom_1", "atom_2"], "tv": {"strength": 0.9, "confidence": 0.8}})");
    
    std::cout << "  ✓ Stored " << storage.size() << " items" << std::endl;
    
    // Test retrieval
    std::string cat_data = storage.retrieve("atom_1");
    std::cout << "  ✓ Retrieved cat data: " << cat_data.substr(0, 50) << "..." << std::endl;
    
    // List all keys
    auto keys = storage.list_keys();
    std::cout << "  ✓ Storage contains " << keys.size() << " keys" << std::endl;
    
    // Validate storage integrity
    if (!storage.validate_storage()) {
        std::cerr << "  FAILED: Storage validation" << std::endl;
        return 1;
    }
    
    std::cout << "  ✅ Storage backend test PASSED" << std::endl;
    return 0;
}
