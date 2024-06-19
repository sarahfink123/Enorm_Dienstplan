#IMPORTS
import streamlit as st
import pandas as pd
import numpy as np
from ortools.sat.python import cp_model
import io
from datetime import datetime


#TITLE
'''
# Dienstplan erstellen
'''

#USER INPUT
c1 = st.columns(2)
month = c1[0].selectbox('Monat', [1,2,3,4,5,6,7,8,9,10,11,12], index=datetime.now().month)
year = c1[1].number_input('Jahr', 2010, 2040, datetime.now().year)
#Upload file
input_df = st.file_uploader('Lade Mitarbeitenden Informationen hoch:', ['XLS', 'XLSX','csv'], help='Es werden nur Dateitypen XLS, XLSX and csv akzeptiert.')
st.caption('Die Mitarbeitendeninformationen sollten eine Spalte haben mit "Mitarbeitenden_ID". Darauf basierend werden die Mitarbeitenden in den Dienstplan eingesteilt.')

#CREATE SCHEDULE
if input_df is not None:
    with st.spinner('Dienstplan wird erstellt...'):
        #Load Data from user input CODE
        if input_df.name.endswith('.csv'):
            mitarbeitende_df = pd.read_csv(input_df)
        elif input_df.name.endswith('.xls') or input_df.name.endswith('.xlsx'):
            mitarbeitende_df = pd.read_excel(input_df)
        else:
            st.error('Unsupported file type. Please upload a CSV, XLS, or XLSX file.')
            st.stop()

        #Operation CODE
        if month in [1, 3, 5, 7, 8, 10, 12]:
            num_days = 31
        elif month in [4, 6, 9, 11]:
            num_days = 30
        elif month == 2:
            num_days = 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28
        else:
            raise ValueError("Invalid month")

        # Generate the date range
        dates = pd.date_range(start=f'{year}-{month:02d}-01', periods=num_days)

        # Declare parameters
        workers = len(mitarbeitende_df['Mitarbeitenden_ID'])
        shifts = 6 # not sure if i can change that
        days = len(dates.strftime('%Y-%m-%d').tolist())
        min_break = 6 # can't really be changed (i dont know why now)
        maxshiftsperday = 1
        maxdifference = 1 #difference between number of shifts per employee (needs to be changed later when hours are taken into consideration)

        # Dictionary to map shift types to time ranges
        shift_types = {
            'Früh 1': '06:00-14:00',
            'Früh 2': '06:00-14:00',
            'Spät 1': '14:00-22:00',
            'Spät 2': '14:00-22:00',
            'Nacht 1': '22:00-06:00',
            'Nacht 2': '22:00-06:00'
        }

        # Initialize model
        model = cp_model.CpModel()

        # Create shift options
        shiftoptions = {}
        for x in range(days):
            for y in range(shifts):
                for z in range(workers):
                    shiftoptions[(x, y, z)] = model.NewBoolVar(f"shift_with_id_{x}_{y}_{z}")

        # Constraint: Each shift is assigned to exactly one worker
        for x in range(days):
            for y in range(shifts):
                model.Add(sum(shiftoptions[(x, y, z)] for z in range(workers)) == 1)

        # Constraint: Each worker works at most one shift per day
        for x in range(days):
            for z in range(workers):
                model.Add(sum(shiftoptions[(x, y, z)] for y in range(shifts)) <= maxshiftsperday)

        # Add constraint ensuring at least a min_break of 6 shifts between consecutive shifts for each worker
        # Add custom break constraints
        for z in range(workers):
            for x in range(days):
                for y in range(shifts):
                    for k in range(1, min_break):  # Adjust based on the break_time
                        if y + k < shifts:
                            model.Add(shiftoptions[(x, y, z)] + shiftoptions[(x, y + k, z)] <= 1)
                        if x + 1 < days and y - k >= 0:
                            model.Add(shiftoptions[(x, y, z)] + shiftoptions[(x + 1, y - k, z)] <= 1)

        # Constraint: Balance the number of shifts per worker
        minshiftsperworker = (shifts * days) // workers
        maxshiftsperworker = minshiftsperworker + maxdifference
        for z in range(workers):
            shiftsassigned = 0
            for x in range(days):
                for y in range(shifts):
                    shiftsassigned += shiftoptions[(x, y, z)]
            model.Add(minshiftsperworker <= shiftsassigned)
            model.Add(shiftsassigned <= maxshiftsperworker)

        # Custom solution printer to store results in a DataFrame
        class SolutionPrinterClass(cp_model.CpSolverSolutionCallback):
            def __init__(self, shiftoptions, workers, days, shifts, sols):
                cp_model.CpSolverSolutionCallback.__init__(self)
                self._shiftoptions = shiftoptions
                self._workers = workers
                self._days = days
                self._shifts = shifts
                self._solutions = set(sols)
                self._solution_count = 0
                self._results = []

            def on_solution_callback(self):
                if self._solution_count in self._solutions:
                    for x in range(self._days):
                        for y in range(self._shifts):
                            for z in range(self._workers):
                                if self.Value(self._shiftoptions[(x, y, z)]):
                                    # Get the shift type and corresponding time range
                                    shift_type = list(shift_types.keys())[y]
                                    shift_time = shift_types[shift_type]
                                    formatted_date = dates[x].strftime('%Y-%m-%d')
                                    self._results.append((dates[x], shift_type, shift_time, mitarbeitende_df['Mitarbeitenden_ID'].iloc[z]))
                self._solution_count += 1

            def get_dataframe(self):
                return pd.DataFrame(self._results, columns=['Datum', 'Schicht', 'Schichtzeit', 'Dienst'])


        # Solve the model
        solver = cp_model.CpSolver()
        solver.parameters.linearization_level = 0
        solutionrange = range(1)  # Display 1 feasible result (the first one)
        solution_printer = SolutionPrinterClass(shiftoptions, workers, days, shifts, solutionrange)
        solver.Solve(model, solution_printer)

        # Get and display the results as a DataFrame
        df = solution_printer.get_dataframe()

        #OUTPUT
        df['Datum'] = pd.to_datetime(df['Datum']).dt.date  # Convert to date only
        weekday_map = {
                    0: 'Mo',
                    1: 'Di',
                    2: 'Mi',
                    3: 'Do',
                    4: 'Fr',
                    5: 'Sa',
                    6: 'So'
                }
        df['Wochentag'] = pd.to_datetime(df['Datum']).dt.weekday.map(weekday_map) # Add weekday
        df = df[['Wochentag', 'Datum', 'Schicht', 'Schichtzeit', 'Dienst']] #Reorder columns

        # Save the DataFrame to an Excel file in memory
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Dienstplan')
            #Style excel sheet
            workbook = writer.book
            worksheet = writer.sheets['Dienstplan']
            yellow_format = workbook.add_format({'bg_color': '#FFFF00'})
            for idx, weekday, in enumerate(df['Wochentag']):
                if weekday in ['Sa', 'So']:
                    worksheet.set_row(idx +1, cell_format=yellow_format)
        buffer.seek(0)

        #Download button
        st.download_button(
            label='Download Dienstplan',
            data=buffer,
            file_name=f'Dienstplan_{year}_{month}.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
