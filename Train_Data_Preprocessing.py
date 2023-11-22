import csv
import requests
import time

# URL and parameters
base_url = "https://datasets-server.huggingface.co/rows"
dataset = "imdb"
config = "plain_text"
split = "train"
total_rows = 2000
batch_size = 100

rows_accumulated = []
offset = 11500

while len(rows_accumulated) < total_rows:
    url = f"{base_url}?dataset={dataset}&config={config}&split={split}&offset={offset}&length={batch_size}"
    response = requests.get(url)
    print(response)

    if response.status_code == 200:
        data = response.json()
        rows = data.get('rows', [])

        if not rows:
            break

        rows_accumulated.extend(rows)
        offset += batch_size
        time.sleep(5)  # Introducing a 1-second delay between requests
    elif response.status_code == 500:
        print(response.content)  # Print server response for debugging
        time.sleep(5)  # Adjust delay as needed and retry
    else:
        print("Failed to fetch the data. Status code:", response.status_code)
        break

# Write a CSV file incrementally within the loop
if rows_accumulated:
    with open("Train_sentiment_dataset.csv", "w", newline='', encoding='utf-8') as csvfile:
        fieldnames = ['text', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows_accumulated[:total_rows]:
            text = row.get('row', {}).get('text', '')
            label = row.get('row', {}).get('label', '')
            writer.writerow({'text': text, 'label': label})

    print("CSV file created successfully!")
else:
    print("No data fetched.")