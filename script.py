from pypdf import PdfReader
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

pdf_file_path = Path('edu.pdf')

def extract_text_from_pdf(path_to_pdf):
    reader = PdfReader(path_to_pdf)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text()
    return full_text

#pdftext = extract_text_from_pdf(pdf_file_path)

'''
with open('pdftext.txt', 'w') as fwrite:
    fwrite.write(pdftext)
    fwrite.close()
'''

with open('pdftext.txt', 'r') as fread:
    pdftext = fread.read()
    fread.close()

model_name = "deepset/roberta-base-squad2"

qa_model = pipeline("question-answering", model=model_name, tokenizer=model_name)
#question = "Identify one or more specific connections or applications of the legislation or policy to the Junior division?"
#{'score': 0.7095813751220703, 'start': 274333, 'end': 274364, 'answer': 'appropriate community resources'}
#{'score': 0.4485955536365509, 'start': 122870, 'end': 122890, 'answer': 'Depression is common'}

question = "Choose one piece of Ontario legislation or a policy that has an impact on Junior classrooms?"
# {'score': 0.5972411632537842, 'start': 17676, 'end': 17708, 'answer': 'Regulated Health Professions Act'}
context = pdftext.replace('\n', ' ')
answer = qa_model(question=question, context=context)

print(answer)


'''
tokenizer = AutoTokenizer.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")
model = AutoModelForSeq2SeqLM.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")
context = pdftext.replace('\n', ' ')


inputs = tokenizer(context, return_tensors="pt")
print(len(inputs))

outputs = model.generate(**inputs, max_length=100)
question_answer = tokenizer.decode(outputs[0], skip_special_tokens=False)
question_answer = question_answer.replace(tokenizer.pad_token, "").replace(tokenizer.eos_token, "")
question, answer = question_answer.split(tokenizer.sep_token)
'''

#print(full_text)
#print(reader.shape)