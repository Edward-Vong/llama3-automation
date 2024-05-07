import ollama
from sentence_transformers import SentenceTransformer, util
from collections import Counter

model = SentenceTransformer('all-MiniLM-L6-v2')  # Load the model for semantic comparisons

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

def calculate_word_overlap(response, correct_answer):
    """ Calculate the normalized word overlap between the response and the correct answer. """
    response_words = Counter(response.lower().split())
    correct_words = Counter(correct_answer.lower().split())
    common_words = response_words & correct_words  # Intersection of both counts
    overlap = sum(common_words.values()) / float(max(len(correct_answer.split()), 1))
    return overlap

def enhanced_semantic_comparison(responses, correct_answers):
    """ Compare responses to the correct answers using both semantic similarity and word overlap. """
    results = []
    correct_count = 0
    for response, correct_answer in zip(responses, correct_answers):
        response_embedding = model.encode(response)
        correct_answer_embedding = model.encode(correct_answer)
        similarity = util.pytorch_cos_sim(response_embedding, correct_answer_embedding).item()
        word_overlap = calculate_word_overlap(response, correct_answer)
        is_correct = similarity > 0.7 and word_overlap > 0.4  # Adjusted thresholds
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

    print(f"LLaMa 3\nSummary: {correct_count} out of {total_questions} questions were answered correctly.\nAccuracy: {accuracy:.2f}%")
