import streamlit as st
import sqlite3
import subprocess

# Create a connection to the SQLite database
conn = sqlite3.connect('users.db')
c = conn.cursor()

# Create table for users if not exists
c.execute('''CREATE TABLE IF NOT EXISTS users (
             username TEXT PRIMARY KEY,
             password TEXT)''')
conn.commit()

def create_user(username, password):
    c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
    conn.commit()

def login(username, password):
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    if c.fetchone():
        return True
    else:
        return False

def open_next_page():
    subprocess.Popen(["streamlit", "run", "TheBigBull.py"])

def register():
    st.title("Registration")
    username = st.text_input("Enter username")
    password = st.text_input("Enter password", type='password')

    if st.button("Register"):
        create_user(username, password)
        st.success("Registration successful. Please log in.")

def main():
    st.header('The BigBull:violet[ Stock Price Prediction]')
    menu = ["Login", "Register"]
    choice = st.selectbox("Menu", menu)

    if choice == "Login":
        st.title("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        
        if st.button("Login"):
            if login(username, password):
                st.success("Logged in as {}".format(username))
                open_next_page()  # Open TTheBigBull.py after successful login
            else:
                st.error("Invalid username or password")

    elif choice == "Register":
        register()

if __name__ == "__main__":
    main()