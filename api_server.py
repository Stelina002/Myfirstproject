import joblib
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

def generate_response(prompt, tokenizer, model, max_length=200, delay=0.2):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.2,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output

def main():
    # Load the trained classifier
    try:
        clf = joblib.load("outputs/fine_tuned_classifier_best.joblib")
    except FileNotFoundError:
        print("❌ Model file not found. Make sure 'fine_tuned_classifier_best.joblib' exists in the 'outputs' folder.")
        return

    # Load the sentence transformer model
    encoder = SentenceTransformer("outputs/sentence_transformer_model")

    # Load the LLM and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
    llm_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM3-3B")

    # Initialize memory and ratings
    conversation_history = {}
    ratings = []
    turn_counter = 1

    # Define personality instructions
    personality_instructions = {
        "friendly": "Respond in a warm and friendly tone.",
        "formal": "Respond in a professional and formal tone.",
        "sarcastic": "Respond with a sarcastic and witty tone.",
        "enthusiastic": "Respond with high energy and excitement."
    }

    # Ask user to choose a personality
    print("👋 Hello! I'm your assistant. Let's get started!")
    print("🧠 Choose a personality for the bot:")
    print("   1. Friendly\n   2. Formal\n   3. Sarcastic\n   4. Enthusiastic")
    while True:
        choice = input("Enter 1-4: ").strip()
        if choice in {"1", "2", "3", "4"}:
            personality = ["friendly", "formal", "sarcastic", "enthusiastic"][int(choice) - 1]
            break
        else:
            print("❗ Please enter a valid number (1-4).")

    print(f"✅ Personality set to: {personality.capitalize()}")
    print("🟢 Type 'exit' or 'quit' to stop.")

    while True:
        user_input = input("\nPrompt: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Thanks for chatting! Have a great day!")
            break
        if not user_input:
            continue

        # Classify input
        embedding = encoder.encode([user_input])
        prediction = clf.predict(embedding)[0]
        label = "blocked" if prediction == 1 else "allowed"
        print("🔍 Prediction:", label)

        if label == "allowed":
            # Build context from last 5 turns
            context = ""
            for i in range(max(1, turn_counter - 5), turn_counter):
                turn = conversation_history.get(i)
                if turn:
                    context += f"User: {turn['user']}\nBot: {turn['bot']}\n"
            context += f"User: {user_input}\nBot:"

            # Add personality instruction
            context = personality_instructions[personality] + "\n" + context

            print("🤖 Generating response...")
            response = generate_response(context, tokenizer, llm_model)

            # Simulate streaming output
            print("💬 Response:", end=" ", flush=True)
            for word in response.split():
                print(word, end=" ", flush=True)
                time.sleep(0.2)
            print("\n✅ Done.\n")

            # Save to memory
            conversation_history[turn_counter] = {"user": user_input, "bot": response}
            turn_counter += 1

            # Ask for rating
            rating = input("⭐ Rate this response (1-5): ").strip()
            ratings.append((user_input, response, rating))

if __name__ == "__main__":
    main()
