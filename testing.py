import tkinter as tk
from tkinter import scrolledtext, Button, Label
import threading
import time
import spacy
from sentence_transformers import SentenceTransformer, util
import ollama

# Initialize the NLP and sentence transformer models
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_ollama_response(question):
    """ Query the ollama chat model for a single question and return response. """
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': question}])
    return response['message']['content']

def extract_main_assertion(text):
    """ Extract the main verb and its direct object as the assertion (simplified). """
    doc = nlp(text)
    for token in doc:
        if token.dep_ in ('ROOT', 'ccomp'):
            return ' '.join([child.text.lower() for child in token.subtree if child.dep_ in ('dobj', 'attr', 'prep')])
    return text.lower()

def enhanced_semantic_comparison(response, correct_answer):
    """ Compare a single response to the correct answer focusing on key assertions. """
    response_assertion = extract_main_assertion(response)
    correct_answer_assertion = extract_main_assertion(correct_answer)
    similarity = util.pytorch_cos_sim(model.encode(response_assertion), model.encode(correct_answer_assertion)).item()
    is_correct = similarity > 0.75
    result = 'Correct' if is_correct else 'Incorrect'
    return response, correct_answer, result, is_correct

def add_chat(question, response):
    """ Add the question and response to the chat interface. """
    chat.config(state=tk.NORMAL)
    chat.insert(tk.END, f"Q: {question}\n", 'question')
    chat.insert(tk.END, f"A: {response}\n\n", 'response')
    chat.config(state=tk.DISABLED)
    chat.yview(tk.END)

def console_log(question, correct_answer, result):
    """ Log the question, expected answer, and result to the console. """
    print(f"Question: {question}\nExpected Answer: {correct_answer}\nResult: {result}")
    print('-' * 80)

def read_file(file_path):
    """ Read text from a file, treating each paragraph separated by an empty line as a separate entry. """
    with open(file_path, 'r', encoding='utf-8') as file:
        entries = []
        current_entry = []
        for line in file:
            stripped_line = line.strip()
            if stripped_line:
                current_entry.append(stripped_line)
            elif current_entry:
                entries.append(" ".join(current_entry))
                current_entry = []
        if current_entry:  # Add the last entry if the file does not end with an empty line
            entries.append(" ".join(current_entry))
        return entries

def process_questions():
    """ Automatically process a list of questions. """
    questions = read_file('questions.txt')
    correct_answers = read_file('answers.txt')  # Assuming this exists
    correct_count = 0
    for question, correct_answer in zip(questions, correct_answers):
        current_question_var.set(question)  # Display the question
        root.update()  # Force GUI update
        time.sleep(1)  # Short pause with question displayed

        current_question_var.set("")  # Clear the question before showing response
        root.update()  # Force update after clearing question
        time.sleep(0.5)  # Short pause before showing response

        response = get_ollama_response(question)
        _, _, result, is_correct = enhanced_semantic_comparison(response, correct_answer)
        add_chat(question, response)
        console_log(question, correct_answer, result)
        if is_correct:
            correct_count += 1
        root.update()  # Update the GUI to show changes
        time.sleep(1)  # Pause for 1 second between questions

    print(f"Summary: {correct_count} out of {len(questions)} questions were answered correctly. Accuracy: {correct_count / len(questions) * 100:.2f}%")

# Set up the main window
root = tk.Tk()
root.title("LLaMa Semantic Chat Interface")
root.geometry("800x700")

# Chat display area
chat = scrolledtext.ScrolledText(root, state=tk.DISABLED, bg="#f0f0f0", bd=0, font=('Arial', 12), wrap=tk.WORD)
chat.tag_configure('question', foreground="#0084FF", justify='left')
chat.tag_configure('response', foreground="#58D68D", justify='left')
chat.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

# Current question display
current_question_var = tk.StringVar(root)
current_question_label = Label(root, textvariable=current_question_var, font=('Arial', 12))
current_question_label.pack(padx=20, pady=10)

# Button to start processing questions
start_auto_button = Button(root, text="Start Processing Questions", command=lambda: threading.Thread(target=process_questions).start())
start_auto_button.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=10)

root.mainloop()
