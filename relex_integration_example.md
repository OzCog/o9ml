# RelEx Integration Example

This document demonstrates the successful integration and testing of the RelEx dependency relation extractor component in the OpenCog Natural Language Cognition layer.

## Component Status: ✅ WORKING

RelEx is successfully running via Docker container and provides:

- **Dependency Parsing**: Extracts grammatical relationships between words
- **Semantic Analysis**: Identifies parts of speech, tense, number, etc.
- **Multiple Output Formats**: Plain text, OpenCog format, Stanford dependencies

## Example Usage

### Input Sentence
```
"Alice ate the mushroom."
```

### RelEx Output

#### Dependency Relations
```
_obj(eat, mushroom)
_subj(eat, Alice)
```

#### Grammatical Attributes
```
pos(., punctuation)
noun_number(mushroom, uncountable)
definite-FLAG(mushroom, T)
pos(mushroom, noun)
penn-POS(mushroom, NN)
pos(the, det)
penn-POS(the, DT)
noun_number(Alice, singular)
definite-FLAG(Alice, T)
gender(Alice, feminine)
pos(Alice, noun)
person-FLAG(Alice, T)
penn-POS(Alice, NN)
pos(eat, verb)
tense(eat, past)
penn-POS(eat, VBD)
```

#### Stanford-Style Dependencies
```
det(mushroom-4-NN, the-3-DT)
nsubj(ate-2-VBD, Alice-1-NN)
dobj(ate-2-VBD, mushroom-4-NN)
```

## Integration with OpenCog Ecosystem

RelEx serves as a critical component in the natural language processing pipeline:

1. **Input**: Raw text sentences
2. **Processing**: Link Grammar parsing + RelEx semantic analysis
3. **Output**: Structured dependency relations and grammatical features
4. **Integration**: Ready for AtomSpace storage and PLN reasoning

## Testing Commands

### Basic Text Processing
```bash
echo "This is a test sentence!" | docker run --rm -i opencog/relex /home/Downloads/relex-master/relation-extractor.sh -n 4 -l -t -f -r -a
```

### Server Mode (Plain Text)
```bash
docker run --rm -p 3333:3333 opencog/relex /home/Downloads/relex-master/plain-text-server.sh
```

### Server Mode (OpenCog Format)
```bash
docker run --rm -p 4444:4444 opencog/relex /home/Downloads/relex-master/opencog-server.sh
```

## Next Steps for Language Layer

With RelEx successfully validated, the next steps are:

1. ✅ Build and test relex component (COMPLETED)
2. Create integration examples with AtomSpace
3. Create integration examples with PLN 
4. Document language tensor shapes and dimensions
5. Create comprehensive language layer tests

## Component Dependencies

- ✅ Link Grammar (syntactic parsing)
- ✅ WordNet (semantic knowledge)
- ✅ OpenNLP (natural language processing)
- ✅ Docker (containerized deployment)