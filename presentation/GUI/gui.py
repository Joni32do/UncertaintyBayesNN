import PySimpleGUI as sg

sg.theme("DarkAmber")

layout = [  [sg.Text('Some text in row 1')],
            [sg.Text('Enter someting on Row 2'),sg.InputText()],
            [sg.Button('OK'), sg.Button('Cancel')]]

window = sg.Window('Mace',layout)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel':
        break
    print(f"You entered {values[0]}")


window.close()

event, values = sg.Window('Get filename example', [[sg.Text('Filename')], [sg.Input(), sg.FileBrowse()], [sg.OK(), sg.Cancel()] ]).read(close=True)