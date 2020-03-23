# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:44:08 2020

@author: -
"""

#import requests
#response = requests.get('http://example.com')
#
#response.request.url
#
#response.request.headers
#
#dir(response.request)
#
#response.status_code
#
#response.headers
#
#response.headers['Server']
#
#response.encoding
#
#response.cookies
#
#response.text # the Payload of the HTTP response, in this case containing the HTML of the page
#
#response = requests.post('http://httpbin.org/post', data = {'OP':'Otago Polytechnic'})
#response.status_code
#
#response.text
#
#response = requests.put('http://httpbin.org/put', data = {'key':'value'}) #Returns PUT data.
#response = requests.delete('http://httpbin.org/delete') #/delete Returns DELETE data
#response = requests.head('http://httpbin.org/get')
#response = requests.options('http://httpbin.org/get')
#
#payload = {'key1': 'value1', 'key2': 'value2'}
#r = requests.get('http://httpbin.org/get', params=payload)
#
#print(r.url)
#
#url = 'http://httpbin.org/put'
#headersDictionary = {'user-agent': 'I\'m a fake browser'}
#
#r = requests.get(url, headers=headersDictionary)
#r.request.headers
#
#payload = {'key1': 'value1', 'key2': 'value2'}
#
#response = requests.post("http://httpbin.org/post", data=payload)
#print(r.text)
#
#from PIL import Image
#from io import BytesIO
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#
#response = requests.get('https://i.ytimg.com/vi/kZw-jgCRPeE/maxresdefault.jpg')
#
#binaryImage = Image.open(BytesIO(response.content))
#binaryImage.save('./imageRetrievedFromTheWeb.png')
#imgplot = plt.imshow(binaryImage)
#plt.axis('off')
#plt.show()

import requests

response = requests.get('http://api.openweathermap.org/data/2.5/weather?q=Dunedin,nz&appid=f6b6fecf2c4292d8d19d201e57667588&mode=json')
response.json()