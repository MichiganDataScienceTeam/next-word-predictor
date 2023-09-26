import regex as re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize

class Process:
    @staticmethod
    def remove_whitespace(text: str) -> str:
        text = re.sub(r'\n+', ' ', text).strip()
        return re.sub(r'\t\r', '', text)

    @staticmethod
    def remove_misc(text: str) -> str:
        text = re.sub(r"“|”|‘|’", '', text)
        return text
    
    @staticmethod
    def remove_basic_roman_numerals(text: str) -> str:
        roman_numeral_str = r"(XI{0,2}\.)|(VI{0,3}\.)|(IV|IX|I{1,3}\.)"

        return re.sub(roman_numeral_str, '', text)
    
    # Helper func to include adjacent punctuation in token
    @staticmethod
    def modify_tokens(tokens):
        modified_tokens = []
        temp_token = ""
        for token in tokens:
            if token.isalnum():  # Check if token is alphanumeric
                if temp_token:
                    modified_tokens.append(temp_token)
                    temp_token = ""
                modified_tokens.append(token)
            else:
                if modified_tokens:
                    temp_token = modified_tokens.pop() + token
                else:
                    temp_token = token
        if temp_token:
            modified_tokens.append(temp_token)
        return modified_tokens

    @staticmethod
    def file_to_sentences(FILE_PATH, remove_whitespace=True, remove_misc=True, remove_basic_roman_numerals=True):
        with open('../data/' + FILE_PATH, 'r', encoding='utf-8') as f:
            try: text = f.read()
            except UnicodeDecodeError: raise Exception('Error in reading file.')
            if remove_whitespace:
                text = Process.remove_whitespace(text)
            if remove_misc:
                text = Process.remove_misc(text)
            if remove_basic_roman_numerals:
                text = Process.remove_basic_roman_numerals(text)

        return sent_tokenize(text)