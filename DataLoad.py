import csv
import requests

url = "https://datasets-server.huggingface.co/rows?dataset=tweet_eval&config=sentiment&split=train&offset=100&length=100"

response = requests.get(url)

if response.status_code == 200:
    data = response.json()

    rows = data.get('rows', [])  # Extracting 'rows' data from the JSON

    with open("tweet_eval_sentiment_subset.csv", "w", newline='', encoding='utf-8') as csvfile:
        fieldnames = ['text', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in rows:
            text = row.get('row', {}).get('text', '')  # Extracting 'text' from each row
            label = row.get('row', {}).get('label', '')  # Extracting 'label' from each row
            writer.writerow({'text': text, 'label': label})
    print("CSV file created successfully!")
else:
    print("Failed to fetch the data. Status code:", response.status_code)
