import ollama
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import spacy

model = SentenceTransformer('all-MiniLM-L6-v2')  # Load the model for semantic comparisons
nlp = spacy.load("en_core_web_sm")  # Load a SpaCy model for NLP tasks

def read_file(file_path):
    """ Read multi-line entries from a text file separated by empty lines. """
    entries = []
    current_entry = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():
                current_entry.append(line.strip())
            elif current_entry:
                entries.append(" ".join(current_entry))
                current_entry = []
        if current_entry:
            entries.append(" ".join(current_entry))
    return entries

def get_ollama_response(question):
    """ Query the ollama chat model for a single question and return response. """
    response = ollama.chat(model='llama3', messages=[
        {'role': 'user', 'content': question}
    ])
    return response['message']['content']

def extract_main_assertion(text):
    """ Extract the main verb and its direct object as the assertion (simplified). """
    doc = nlp(text)
    for token in doc:
        if token.dep_ in ('ROOT', 'ccomp'):
            return ' '.join([child.text.lower() for child in token.subtree if child.dep_ in ('dobj', 'attr', 'prep')])
    return text.lower()  # Fallback to the original text if specific extraction fails

def enhanced_semantic_comparison(response, correct_answer):
    """ Compare a single response to the correct answer focusing on key assertions. """
    response_assertion = extract_main_assertion(response)
    correct_answer_assertion = extract_main_assertion(correct_answer)
    similarity = util.pytorch_cos_sim(model.encode(response_assertion), model.encode(correct_answer_assertion)).item()
    is_correct = similarity > 0.75  # Adjusted threshold
    result = ('Correct' if is_correct else 'Incorrect')
    return response, correct_answer, result, is_correct

if __name__ == "__main__":
    questions_file = 'questions.txt'
    answers_file = 'answers.txt'

    questions = read_file(questions_file)
    correct_answers = read_file(answers_file)
    correct_count = 0

    for question, correct_answer in zip(questions, correct_answers):
        response = get_ollama_response(question)
        response, correct_answer, result, is_correct = enhanced_semantic_comparison(response, correct_answer)
        print(f"Question: {question}\nResponse: {response}\nCorrect Answer: {correct_answer}\nResult: {result}\n")
        print('-' * 80)
        if is_correct:
            correct_count += 1

    total_questions = len(questions)
    accuracy = (correct_count / total_questions) * 100
    print(f"LLaMa 3\nSummary: {correct_count} out of {total_questions} questions were answered correctly.\nAccuracy: {accuracy:.2f}%")
