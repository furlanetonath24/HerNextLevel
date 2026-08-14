import gradio as gr
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import torch

# This is the same pattern from the Generative AI lesson! It uses the
# Inference Provider API to send your messages to an AI model and get
# a response back. Swap out the model below for a different one if
# you want to experiment!
#
# Note: if this Space doesn't already have one, you'll need to add an
# HF_TOKEN secret in the Space's Settings tab for this to work
# (Settings -> Variables and secrets -> New secret).

client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", bill_to="kode-with-klossy")


with open("knowledge.txt", "r", encoding="utf-8") as file:
  # Read the entire contents of the file and store it in a variable
    knowledge_txt = file.read()
with open("career_knowledge.txt", "r", encoding="utf-8") as f:
    career_knowledge = f.read()

def preprocess_text(text):
  # Strip extra whitespace from the beginning and the end of the text
  cleaned_text = text.strip()

  # Split the cleaned_text by every newline character (\n)
  chunks = cleaned_text.split("\n\n")

  # Create an empty list to store cleaned chunks
  cleaned_chunks = []

  # Write your for-in loop below to clean each chunk and add it to the cleaned_chunks list
  # This is only one way scholars may write this, but there are other ways!
  for chunk in chunks:
    stripped_chunk = chunk.strip()
    if len(stripped_chunk) > 0:
      cleaned_chunks.append(stripped_chunk)

  # ===== SPICY CHALLENGE: LIST COMPREHENSION =====
  # The if chunk.strip() conditional is truthy if the string is not empty
  # cleaned_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

  # Print cleaned_chunks
  print(cleaned_chunks)

  # Print the length of cleaned_chunks
  print(len(cleaned_chunks))

  # Return the cleaned_chunks
  return cleaned_chunks

# Call the preprocess_text function and store the result in a cleaned_chunks variable
cleaned_chunks = preprocess_text(knowledge_txt)
                                
# Load the pre-trained embedding model that converts text to vectors
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(text_chunks):
  # Convert each text chunk into a vector embedding and store as a tensor
  chunk_embeddings = model.encode(text_chunks, convert_to_tensor=True) # Replace ... with the cleaned_chunks list

  # Print the chunk embeddings
  print(chunk_embeddings)

  # Print the shape of chunk_embeddings
  print(chunk_embeddings.shape)

  # Return the chunk_embeddings
  return chunk_embeddings

# Call the create_embeddings function and store the result in a new chunk_embeddings variable
chunk_embeddings = create_embeddings(cleaned_chunks) # Complete this line)

# Define a function to find the most relevant text chunks for a given query, chunk_embeddings, and text_chunks
def get_top_chunks(query, chunk_embeddings, text_chunks):
  # Convert the query text into a vector embedding
  query_embedding = model.encode(query, convert_to_tensor=True) # Complete this line

  # Normalize the query embedding to unit length for accurate similarity comparison
  query_embedding_normalized = query_embedding / query_embedding.norm()

  # Normalize all chunk embeddings to unit length for consistent comparison
  chunk_embeddings_normalized = chunk_embeddings / chunk_embeddings.norm(dim=1, keepdim=True)

  # Calculate cosine similarity between all chunks and the query using matrix multiplication
  similarities = torch.matmul(chunk_embeddings_normalized, query_embedding_normalized) # Complete this line

  # Print the similarities
  print(similarities)

  # Find the indices of the 3 chunks with highest similarity scores
  top_indices = torch.topk(similarities, k=3).indices

  # Print the top indices
  print(top_indices)

  # Create an empty list to store the most relevant chunks
  top_chunks = []

  # Loop through the top indices and retrieve the corresponding text chunks
  # This is only one way scholars may write this, but there are other ways!
  for i in top_indices:
    chunk = text_chunks[i]
    top_chunks.append(chunk)

  # Return the list of most relevant chunks
  return top_chunks
    
def respond(message, history): #respond function
    top_results = get_top_chunks(message, chunk_embeddings, cleaned_chunks) 
    messages = [{"role": "system", "content": f"You are a friendly chatbot. Use the following research context to help answer questions:\n\n{top_results}"}]

    # Safely format history for the client chat_completion API
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    
 #   if history:
 #       messages.extend(history)

    messages.append({"role": "user", "content": message})

    response = client.chat_completion(
        messages,
        max_tokens=1024,
        temperature = 0.7
    )
    # return response.choices.message.content.strip()
    return response.choices[0].message.content.strip()
# =========================
# CAREER QUIZ LOGIC
# =========================
career_matches = {

    "Software Engineer": [
        "computer_science",
        "coding",
        "technology",
        "problem_solving",
        "math",
        "software"
    ],

    "Data Scientist": [
        "computer_science",
        "coding",
        "technology",
        "problem_solving",
        "math",
        "data"
    ],

    "Cybersecurity Analyst": [
        "computer_science",
        "coding",
        "technology",
        "problem_solving"
    ],

    "Mechanical Engineer": [
        "engineering",
        "design",
        "building",
        "physics",
        "math",
        "problem_solving"
    ],

    "Aerospace Engineer": [
        "engineering",
        "space",
        "physics",
        "design",
        "building",
        "math"
    ],

    "Robotics Engineer": [
        "engineering",
        "coding",
        "robotics",
        "design",
        "building",
        "problem_solving",
        "robots"
    ],

    "Biomedical Engineer": [
        "biology",
        "engineering",
        "technology",
        "research",
        "helping_people",
        "problem_solving",
        "medical technology"
    ],

    "Chemical Engineer": [
        "chemistry",
        "engineering",
        "math",
        "research",
        "experiments"
    ],

    "Environmental Engineer": [
        "engineering",
        "environment",
        "science",
        "problem_solving",
        "design",
        "environmental solutions"
    ],

    "Astrophysicist": [
        "space",
        "physics",
        "math",
        "research",
        "science"
    ],

    "Research Scientist": [
        "biology",
        "chemistry",
        "science",
        "research",
        "experiments"
    ],

    "Materials Scientist": [
        "chemistry",
        "physics",
        "engineering",
        "research",
        "materials",
        "new materials"
    ],

    "Civil Engineer": [
        "engineering",
        "design",
        "building",
        "physics",
        "math",
        "problem_solving"
    ],

    "Electrical Engineer": [
        "engineering",
        "physics",
        "technology",
        "electronics",
        "math",
        "problem_solving"
    ],

    "Computer Engineer": [
        "computer_science",
        "coding",
        "engineering",
        "technology",
        "electronics",
        "problem_solving"
    ],

    "Chemical Research Scientist": [
        "chemistry",
        "research",
        "experiments",
        "science",
        "materials"
    ],

    "Environmental Scientist": [
        "biology",
        "environment",
        "science",
        "research",
        "experiments",
        "environmental solutions"
    ],

    "Biomedical Scientist": [
        "biology",
        "chemistry",
        "research",
        "science",
        "experiments",
        "medical technology"
    ],

    "Nuclear Engineer": [
        "engineering",
        "physics",
        "math",
        "science",
        "energy",
        "problem_solving"
    ],

    "Industrial Engineer": [
        "engineering",
        "math",
        "data",
        "problem_solving",
        "technology",
        "design"
    ]
}


def career_quiz(q1, q2, q3, q4, q5, q6):

    answers = [q1, q2, q3, q4, q5, q6]

    scores = {}

    for career, interests in career_matches.items():

        score = 0

        for answer in answers:

            if answer in interests:
                score += 1

        scores[career] = score

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_careers = ranked[:3]

    result = "## 🌟 Your Top STEM Career Matches\n\n"

    for i, (career, score) in enumerate(top_careers, start=1):

        result += f"### {i}. {career}\n"
        result += f"**Match:** {score}/6\n\n"
        
        # Search the career knowledge base
        career_position = career_knowledge.lower().find(career.lower())

        if career_position != -1:

            career_info = career_knowledge[career_position:]

            # Limit how much information is displayed
            career_info = career_info[:1200]

            result += career_info
            result += "\n\n---\n\n"

    return result



with gr.Blocks(theme=gr.Theme.from_hub("Ayaku/Aa")) as demo:
    # Custom HTML Banner
    gr.Markdown(
        """
        <div style="text-align: center; padding: 25px; background: linear-gradient(135deg, #6C5CE7, #A8A5FF); color: white; border-radius: 12px; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h1 style="font-size: 2.6rem; margin: 0; font-family: sans-serif;">👩‍💻 Tech Quest: Her Next Level</h1>
            <p style="font-size: 1.1rem; margin-top: 8px; opacity: 0.9;">Empowering young women in tech with elite opportunities, camps, and custom project guidance.</p>
        </div>
        """
    )
    with gr.Row():
        with gr.Column(scale=3): 
            # ChatInterface with perfectly scrubbed example formatting
            gr.ChatInterface(
            respond,
            examples=[
                ["What tech opportunities can I apply for?"],
                ["What programs are available for girls interested in technology?"],
                ["Give me ideas for a passion project in tech"],
                ["How can I start learning programming?"]
                ],
            description="This Chatbot is for girls interested in tech who want more guidance on opportunities, programs, and passion project ideas. It provides you with resources that you can choose from.",
            )
        with gr.Column(scale=1):

            gr.Markdown("""
            ## 🌟 Career Explorer

            Not sure what STEM career
            is right for you?

            Take our quick quiz!
            """)

            quiz_button = gr.Button("✨ Take the Quiz")
        with gr.Column(visible=False) as quiz_area:

            q1 = gr.Radio(
                [
                    "computer_science",
                    "engineering",
                    "biology",
                    "chemistry",
                    "physics",
                    "space"
                ],
                label="1. Which STEM subject interests you the most?"
                )

            q2 = gr.Radio(
                [
                    "coding",
                    "design",
                    "building",
                    "research",
                    "experiments",
                    "problem_solving"
                ],
                label="2. What kind of problem would you most enjoy solving?"
                )

            q3 = gr.Radio(
                [
                    "coding",
                    "building",
                    "experiments",
                    "data",
                    "design",
                    "exploring"
                ],
                label="3. Which activity sounds the most fun?"
                )

            q4 = gr.Radio(
                [
                    "technology",
                    "laboratory",
                    "workshop",
                    "office",
                    "research",
                    "outdoors"
                ],
                label="4. Where would you most enjoy working?"
                )

            q5 = gr.Radio(
                [
                    "robots",
                    "software",
                    "space",
                    "medical technology",
                    "new materials",
                    "environmental solutions"
                ],
                label="5. What would you most like to create or explore?"
                )

            q6 = gr.Radio(
                [
                    "technology",
                    "helping_people",
                    "research",
                    "environmental solutions",
                    "space",
                    "problem_solving
                ],
                label="6. What impact would you most like your STEM career to have?"
             )

        find_button = gr.Button("🔍 Find My Careers")

        quiz_results = gr.Markdown()

        find_button.click(
            career_quiz,
            inputs=[q1, q2, q3, q4, q5, q6],
            outputs=quiz_results
         )

        quiz_button.click(
        lambda: gr.update(visible=True),
        outputs=quiz_area
         )

demo.launch()


    
#chatbot = gr.ChatInterface(respond, title = "Tech Quest: Her Next Level", description = "This Chatbot is for girls interested in tech who wnat more guidance on opportunities, programs, and passion project ideas. It provides you with resources that you can choose from", banner = "👩‍💻")
#chatbot.launch(theme=gr.Theme.from_hub("Ayaku/Aa"))


# TODO: This is just a starting point! Customize the system prompt,
# the model, and the interface to make this project your own!
