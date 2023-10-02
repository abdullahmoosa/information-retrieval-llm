import streamlit_authenticator as stauth

names = ['BankAI']
usernames = ['bank123']
passwords = ['123']

hashed_passwords = stauth.Hasher(passwords).generate()
print(hashed_passwords)