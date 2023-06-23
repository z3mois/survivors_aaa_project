import streamlit as st
import sys
import pandas as pd
from faker import Faker
sys.path.append('model_utils')
from inference_func import load_vocabs, make_indexes_from_tuple, model_inference_one_vac
from model import create_model
import torch
import phonenumbers
# Define placeholder functions for the neural network and database queries

def get_top_resumes(similarity_scores):
    # Replace this with your database query code
    return [f'{resume}' for resume, _ in similarity_scores]
def get_number(size):
    """
        create fake_number
    """
    faker = Faker('ru_RU')
    phone_numbers = []
    i = 0
    while i < size:
        fake_number = faker.phone_number()
        try:
            parsed_number = phonenumbers.parse(fake_number, 'RU')
            if phonenumbers.is_valid_number(parsed_number):
                formatted_number = phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.E164)
                phone_numbers.append(formatted_number)
                i += 1
        except: 
            pass
    return phone_numbers
# Create the Streamlit app
def main():
    st.set_page_config(page_title="Поиск кандидатов", page_icon=":guardsman:", layout="wide")
    vac_vocab, res_vocab = load_vocabs()
    model = create_model()
    final_state = torch.load('data/final_model_state.pth', map_location='cpu')
    model.load_state_dict(final_state)

    # Create a sidebar with a title and description
    st.sidebar.title("Инструкция")
    st.sidebar.write("Заполните нужные поля и нажмите на кнопку, чтобы найти подходящее резюме.")

    # Create a text input for the vacancy text
    description = st.text_area("Введите описание вакансии:")
    city = st.text_input("Введите город работы")
    top_k = st.text_input("Введите количество резюме, которое хотите увидеть",value="10")
    # Create a button text_input trigger the similarity calculation
    if st.button('Подобрать кандидата'):

        # Show a progress bar while the similarity scores are being calculated
        with st.spinner('Cчитаем похожесть...'):
            data = None
            if not top_k:
                result = model_inference_one_vac(model, vac_vocab, res_vocab, description, city, 'cpu')
                data = pd.DataFrame(result, columns=["Описание резюме", "Похожесть"])
                data = data.drop('Похожесть', axis=1)
                data["Телефон"] = get_number(len(data))
            else:
                try:
                    top_k = int(top_k)
                    if top_k <= 0:
                        st.error(f"Произошла ошибка: Введите корректное число вакансий")
                    else:
                        result = model_inference_one_vac(model, vac_vocab, res_vocab, description, city, 'cpu')
                        data = pd.DataFrame(result, columns=["Описание резюме", "Похожесть"]).iloc[0:top_k]
                        data = data.drop('Похожесть', axis=1)
                        data["Телефон"] = get_number(len(data))
                except Exception as e:
                    st.error(f"Произошла ошибка: Введите корректное число вакансий")
            
            if not data  is None:
                st.table(data)

# Start the app
if __name__ == '__main__':
    main()