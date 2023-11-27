import csv
import requests

# URL and parameters
base_url = "https://datasets-server.huggingface.co/rows"
dataset = "tweet_eval"
config = "sentiment"
split = "train"
total_rows = 2000
batch_size = 100

rows_accumulated = []
offset = 0

while len(rows_accumulated) < total_rows:
    url = f"{base_url}?dataset={dataset}&config={config}&split={split}&offset={offset}&length={batch_size}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        rows = data.get('rows', [])

        if not rows:
            break

        rows_accumulated.extend(rows)
        offset += batch_size
    else:
        print("Failed to fetch the data. Status code:", response.status_code)
        break

# Write a CSV file
if rows_accumulated:
    with open("test_sentiment_dataset.csv", "w", newline='', encoding='utf-8') as csvfile:
        fieldnames = ['text', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in rows_accumulated[:total_rows]:
            label = row.get('row', {}).get('label', '')
            if label in [0,2]:
                if label == 2:
                    label = 1
                text = row.get('row', {}).get('text', '')
                writer.writerow({'text': text, 'label': label})

            # text = row.get('row', {}).get('text', '')
            # label = row.get('row', {}).get('label', '')
            # print(type(label))
            # writer.writerow({'text': text, 'label': label})

    print("CSV file created successfully!")
else:
    print("No data fetched.")



