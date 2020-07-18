import requests, webbrowser
from bs4 import BeautifulSoup
import os 
import sklearn

keyword = input("Enter something to search: ")
#keywords = ["ebola virus","python" , "geeksforgeeks"]


# path of the file
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = dir_path + "/Datasets" #<--- assuming Dataset folder already exists
path = os.path.join(parent_dir, keyword)
os.mkdir(path)

print("Googling....")

# look through 50 google searches to find 10
google_search = requests.get("https://www.google.com/search?q={}&num=50&hl=en".format(keyword))
soup = BeautifulSoup(google_search.text, 'html.parser')
result_div = soup.find_all('div',attrs = {'class':'ZINbbc'})
links = []
titles = []
descriptions = []

# finding the 50 links
for r in result_div:
    # Checks if each element is present, else, raise exception
    try:
        link = r.find('a', href = True)
        title = r.find('div', attrs={'class':'vvjwJb'}).get_text()
        description = r.find('div', attrs={'class':'s3v9rd'}).get_text()
        
        # Check to make sure everything is present before appending
        if link != '' and title != '' and description != '':
            links.append([title,link['href']])
            
            #file.write(description)
    # Next loop if one element is not present
    except:
        continue

# going through the 50 links to find the 10 
num_pages = 10
num_words = 200
count = 0
i = 0
while (count < num_pages):

    # for links that don't start with 'http'
    # it removes the start until the link now starts with 'http'
    if not links[i][1].startswith("http"):
        reached = False
        
        while not reached:
            links[i][1] = links[i][1][1:]
            reached = False if not links[i][1].startswith("http") else True


    # opens the link to read it
    page_search = requests.get(links[i][1])
    page_soup = BeautifulSoup(page_search.content,'html.parser')

    # the amount of text on the page
    length= len(page_soup.get_text().split())

    # will write in document if length of text meets the criteria
    if length > num_words:
        title = links[i][0]
        temp_path = parent_dir+"/"+keyword+"/"
        file_path = os.path.join(temp_path,title)
        file = open(file_path+".txt","w+",encoding="utf8")
        file.write(page_soup.get_text())
        
        count = count+1 # count of # of pages

        file.close()
    i+=1 #count of the links in the array


print("Done Searching")

    

