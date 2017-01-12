import requests

sources = open("TomatoURLs.txt",'r')
img_data = requests.get("http://farm1.static.flickr.com/211/456466127_6bed792d6f.jpg").content
with open('image_name.jpg', 'wb') as handler:
    handler.write(img_data)
