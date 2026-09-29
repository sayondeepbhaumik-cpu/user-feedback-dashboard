import pandas as pd
import random
import os

# Check if the file exists first
if not os.path.exists("users.csv"):
    print("❌ Error: Cannot find users.csv in this directory!")
else:
    # A clean list of names to mix and match randomly
    first_names = ["Rahul", "Priya", "Amit", "Neha", "Rohan", "Anjali", "Vikram", "Sneha", "Aditya", "Tanvi", "Deepak", "Kiran", "Sanjay", "Divya", "Arjun", "Riya"]
    last_names = ["Sharma", "Patel", "Verma", "Gupta", "Joshi", "Mehta", "Singh", "Das", "Sen", "Bhaumik", "Roy", "Kumar", "Nair", "Reddy", "Rao", "Mishra"]

    # 1. Load your existing users file
    df = pd.read_csv("users.csv")

    # 2. Build 60 unique full names
    random.seed(42)
    generated_names = []
    for i in range(len(df)):
        full_name = f"{random.choice(first_names)} {random.choice(last_names)}"
        generated_names.append(full_name)

    # 3. Replace the robotic column with your fresh human names list
    df['name'] = generated_names

    # 4. Save the changes back to your file cleanly
    df.to_csv("users.csv", index=False)
    print("🎉 Success! 60 realistic human names have been generated and saved into users.csv!")