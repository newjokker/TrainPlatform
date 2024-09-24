import requests



res = requests.post("http://192.168.3.69:11202/register_agent", json={"name": "test3", "host": "192.168.3.21", "port": 12345})
print(res.text)

# res = requests.get("http://192.168.3.69:11202/register_info")
# print(res.text)


# res = requests.post("http://192.168.3.69:11202/register_agent", json={"name": "test3", "host": "192.168.3.213", "port": 12345})
# print(res.text)

# res = requests.get("http://192.168.3.69:11202/register_info")
# print(res.text)


# res = requests.post("http://192.168.3.69:11202/register_agent", json={"name": "test4", "host": "192.168.3.231", "port": 12345})
# print(res.text)

# res = requests.get("http://192.168.3.69:11202/register_info")
# print(res.text)

# res = requests.post("http://192.168.3.69:11202/deregister_agent", json={"name": "test4", "host": "192.168.3.231", "port": 12345})
# print(res.text)

# res = requests.get("http://192.168.3.69:11202/register_info")
# print(res.text)







