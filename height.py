import streamlit as st
import pandas as pd
import datetime

# Mock data (replace with real data source)
data = {
    'Date': [datetime.date(2023, 1, 1), datetime.date(2023, 2, 1), datetime.date(2023, 3, 1)],
    'Height (cm)': [50, 55, 60]  # Replace with actual height measurements
}

# Create a DataFrame from mock data
df = pd.DataFrame(data)

# Streamlit app
st.title("Tree or Vegetable Growth Rate Monitor")

# Display the mock data
st.write("Growth Data", df)

# Calculate and display growth rate
def calculate_growth_rate(data_frame):
    if len(data_frame) >= 2:
        initial_height = data_frame.iloc[0]['Height (cm)']
        final_height = data_frame.iloc[-1]['Height (cm)']
        days_passed = (data_frame.iloc[-1]['Date'] - data_frame.iloc[0]['Date']).days
        growth_rate = (final_height - initial_height) / days_passed
        return growth_rate
    else:
        return None

growth_rate = calculate_growth_rate(df)
if growth_rate is not None:
    st.write(f"Growth Rate: {growth_rate:.2f} cm/day")

# Input form to add new height measurements
st.header("Add New Height Measurement")
new_date = st.date_input("Date")
new_height = st.number_input("Height (cm)")

if st.button("Add Measurement"):
    new_row = {'Date': new_date, 'Height (cm)': new_height}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    st.success("Measurement added!")

    # Recalculate and display the updated growth rate
    growth_rate = calculate_growth_rate(df)
    if growth_rate is not None:
        st.write(f"Updated Growth Rate: {growth_rate:.2f} cm/day")

# Display the updated data
st.write("Updated Growth Data", df)

# Note: In a real-world scenario, you would replace the mock data with actual data from sensors or databases.
