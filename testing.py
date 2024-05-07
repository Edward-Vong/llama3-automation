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

def get_ollama_responses(questions):
    """ Query the ollama chat model for each question and return responses. """
    responses = []
    for question in questions:
        response = ollama.chat(model='llama3', messages=[
            {'role': 'user', 'content': question}
        ])
        responses.append(response['message']['content'])
    return responses

def extract_main_assertion(text):
    """ Extract the main verb and its direct object as the assertion (simplified). """
    doc = nlp(text)
    for token in doc:
        if token.dep_ in ('ROOT', 'ccomp'):
            return ' '.join([child.text.lower() for child in token.subtree if child.dep_ in ('dobj', 'attr', 'prep')])
    return text.lower()  # Fallback to the original text if specific extraction fails

def enhanced_semantic_comparison(responses, correct_answers):
    """ Compare responses to the correct answers focusing on key assertions. """
    results = []
    correct_count = 0
    for response, correct_answer in zip(responses, correct_answers):
        response_assertion = extract_main_assertion(response)
        correct_answer_assertion = extract_main_assertion(correct_answer)
        similarity = util.pytorch_cos_sim(model.encode(response_assertion), model.encode(correct_answer_assertion)).item()
        is_correct = similarity > 0.75  # Adjusted threshold
        results.append((response, correct_answer, 'Correct' if is_correct else 'Incorrect'))
        if is_correct:
            correct_count += 1
    return results, correct_count

if __name__ == "__main__":
    questions_file = 'questions.txt'
    answers_file = 'answers.txt'

    questions = read_file(questions_file)
    correct_answers = read_file(answers_file)
    responses = get_ollama_responses(questions)
    results, correct_count = enhanced_semantic_comparison(responses, correct_answers)

    total_questions = len(questions)
    accuracy = (correct_count / total_questions) * 100

    for question, (response, correct_answer, result) in zip(questions, results):
        print(f"Question: {question}\nResponse: {response}\nCorrect Answer: {correct_answer}\nResult: {result}\n")
        print('-' * 80)

    print(f"LLaMa 3\nSummary: {correct_count} out of {total_questions} questions were answered correctly.\nAccuracy: {accuracy:.2f}%")
