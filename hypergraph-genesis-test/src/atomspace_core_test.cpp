//
// AtomSpace Core Test - Real hypergraph operations
//
#include <iostream>
#include <vector>
#include <memory>
#include <cassert>
#include <map>
#include <string>

// Simplified AtomSpace representation for testing
namespace atomspace_test {

struct TruthValue {
    float strength;
    float confidence;
    
    TruthValue(float s = 0.5f, float c = 0.5f) : strength(s), confidence(c) {}
};

class Atom {
public:
    size_t id;
    std::string type;
    std::string name;
    TruthValue tv;
    
    Atom(size_t i, const std::string& t, const std::string& n, const TruthValue& truth)
        : id(i), type(t), name(n), tv(truth) {}
    
    virtual ~Atom() = default;
};

class Node : public Atom {
public:
    Node(size_t id, const std::string& type, const std::string& name, const TruthValue& tv)
        : Atom(id, type, name, tv) {}
};

class Link : public Atom {
public:
    std::vector<size_t> outgoing;
    
    Link(size_t id, const std::string& type, const std::vector<size_t>& out, const TruthValue& tv)
        : Atom(id, type, "", tv), outgoing(out) {}
};

class AtomSpace {
private:
    std::map<size_t, std::unique_ptr<Atom>> atoms;
    size_t next_id;
    
public:
    AtomSpace() : next_id(1) {}
    
    size_t add_node(const std::string& type, const std::string& name, const TruthValue& tv = TruthValue()) {
        size_t id = next_id++;
        atoms[id] = std::make_unique<Node>(id, type, name, tv);
        return id;
    }
    
    size_t add_link(const std::string& type, const std::vector<size_t>& outgoing, const TruthValue& tv = TruthValue()) {
        // Validate outgoing atoms exist
        for (size_t atom_id : outgoing) {
            if (atoms.find(atom_id) == atoms.end()) {
                throw std::runtime_error("Invalid outgoing atom ID: " + std::to_string(atom_id));
            }
        }
        
        size_t id = next_id++;
        atoms[id] = std::make_unique<Link>(id, type, outgoing, tv);
        return id;
    }
    
    Atom* get_atom(size_t id) {
        auto it = atoms.find(id);
        return (it != atoms.end()) ? it->second.get() : nullptr;
    }
    
    std::vector<size_t> get_atoms_by_type(const std::string& type) {
        std::vector<size_t> result;
        for (const auto& pair : atoms) {
            if (pair.second->type == type) {
                result.push_back(pair.first);
            }
        }
        return result;
    }
    
    size_t get_atom_count() const { return atoms.size(); }
    
    bool validate_integrity() {
        std::cout << "  Validating AtomSpace integrity..." << std::endl;
        
        size_t node_count = 0;
        size_t link_count = 0;
        
        for (const auto& pair : atoms) {
            if (dynamic_cast<Node*>(pair.second.get())) {
                node_count++;
            } else if (auto link = dynamic_cast<Link*>(pair.second.get())) {
                link_count++;
                
                // Validate all outgoing atoms exist
                for (size_t outgoing_id : link->outgoing) {
                    if (atoms.find(outgoing_id) == atoms.end()) {
                        std::cerr << "    ERROR: Link " << pair.first 
                                 << " references non-existent atom " << outgoing_id << std::endl;
                        return false;
                    }
                }
            }
        }
        
        std::cout << "    ✓ Found " << node_count << " nodes and " << link_count << " links" << std::endl;
        std::cout << "    ✓ All link references are valid" << std::endl;
        return true;
    }
};

} // namespace atomspace_test

int main() {
    std::cout << "Testing AtomSpace core functionality..." << std::endl;
    
    using namespace atomspace_test;
    
    AtomSpace as;
    
    // Create concept hierarchy
    size_t entity = as.add_node("ConceptNode", "Entity", TruthValue(1.0f, 0.9f));
    size_t animal = as.add_node("ConceptNode", "Animal", TruthValue(0.9f, 0.8f));
    size_t mammal = as.add_node("ConceptNode", "Mammal", TruthValue(0.8f, 0.9f));
    size_t cat = as.add_node("ConceptNode", "Cat", TruthValue(0.95f, 0.9f));
    
    // Create inheritance links
    size_t animal_isa_entity = as.add_link("InheritanceLink", {animal, entity}, TruthValue(0.9f, 0.8f));
    size_t mammal_isa_animal = as.add_link("InheritanceLink", {mammal, animal}, TruthValue(0.8f, 0.9f));
    size_t cat_isa_mammal = as.add_link("InheritanceLink", {cat, mammal}, TruthValue(0.9f, 0.8f));
    
    std::cout << "  ✓ Created " << as.get_atom_count() << " atoms" << std::endl;
    
    // Test pattern matching
    auto concept_nodes = as.get_atoms_by_type("ConceptNode");
    auto inheritance_links = as.get_atoms_by_type("InheritanceLink");
    
    std::cout << "  ✓ Found " << concept_nodes.size() << " ConceptNodes" << std::endl;
    std::cout << "  ✓ Found " << inheritance_links.size() << " InheritanceLinks" << std::endl;
    
    // Validate integrity
    if (!as.validate_integrity()) {
        std::cerr << "  FAILED: AtomSpace integrity validation" << std::endl;
        return 1;
    }
    
    std::cout << "  ✅ AtomSpace core test PASSED" << std::endl;
    return 0;
}
