import json



filename = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/dataset/국토정보플랫폼/400x400array_data.json"
with open(filename, 'r', encoding='utf-8') as file:
    data = json.load(file)

num = 0
for i in data:
    num += 1
    if i[0] == 126.87242 and i[1] == 37.5502:
        print(num)

print()

