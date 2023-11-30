import PySimpleGUI as sg
from model import *

layout = [
    [sg.Text('Ваш запрос: '), sg.InputText(key='input')],
    [sg.Output(size=(88, 20), key='-OUTPUT-')],
    [sg.Submit('Generate'), sg.Cancel()]
]

window = sg.Window('ru_gpt', layout)

while True:  # The Event Loop
    event, values = window.read()
    if event in (None, 'Exit', 'Cancel'):
        break
    if event == 'Generate':
        window['-OUTPUT-'].update('')
        try:
            response = generate_response(values['input'])
            print(response)
        except Exception as e:
            print(f"Произошла ошибка: {e}")

window.close()