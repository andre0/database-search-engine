'''
Complete the Search Engine
Group 30
'''
import pymongo
import sys
import re
from bs4 import BeautifulSoup
from collections import Counter
from urllib.parse import urlparse
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import math

def tokenization(stringlist) -> list:
    """Takes a list of strings, passes NLTK's PorterStemmer, returns a list of tokens"""
    contentset = Counter()
    ps = PorterStemmer()
    stoplist = ['a', 'an', 'the', 'as', 'at', 'be', 'do', 'for', 'had', 'has', 'is', 'it', 'of', 'on']
    for string in stringlist:
        wordlist = word_tokenize(re.sub(r'[\W_\\]+',' ', string)) #Removes punctuation, \'s, and _'s and tokenizes
        for word in wordlist:
            word = ps.stem(word) # Pass through PorterStemmer
            if word not in stoplist:
                contentset[word]  += 1
    return contentset

def main_menu(corpus):
    """Runs the main menu of the program.
    (Used to run the program from this module.)"""
    running = True
    print('Welcome to CS121 / INF141 Search Engine')
    while running:
        cmd = input("""
Enter a command...
D - Delete the current collection
C - Crawl a directory and create a collection
P - Print the current collection
X - Execute a specified command
Q - Quit
Enter - Search the collection for a query
""").strip()
        if cmd.lower() == 'd':
            corpus.DeleteCol() # Great for debugging, but requires a recrawl
        elif cmd.lower() == 'c':
            corpus.CrawlCorpus(corpus.directory)
        elif cmd.lower() == 'p':
            corpus.ShowCol()
        elif cmd.lower() == 'q':
            running = False
        elif cmd.lower() == 'x': # Input a command like you normally would in python. Great for debugging
            while True:
                execcmd = input("Input a command or 'return' to exit: ")
                if execcmd == 'return':
                    break
                else:
                    try:
                        exec(execcmd)
                    except Exception as e:
                        print(e)
        elif cmd == '':
            try:
                with open(corpus.directory+'bookkeeping.json','r') as inf:
                    bookkeeping = eval(inf.read())
                    for result in corpus.SearchCol(input("Please enter a search query: ")):
                        print(bookkeeping[result])
            except (TypeError, IndexError):
                print("No results found")

class Index:
    def __init__(self, directory):
        """Initializes (or re-initializes) the pymongo client used to maintain the token index.
        Requires a directory argument with a valid bookkeeping.json file"""
        self.myclient = pymongo.MongoClient("mongodb://localhost:27017/")
        self.mydb = self.myclient["mydatabase"]
        self.mycol = self.mydb["webpages"]
        self.directory = directory # The directory that will be used to fetch url data
        self.pagecount = self.mydb["webpagecount"] # Attaches a variable to the database that keeps track of # of documents

    def DeleteCol(self):
        """Deletes the collection of tokens, and requires a re-scrape of data. Used mostly for debugging"""
        conf = input('Deleting the current collection. Are you sure? Y/N ')
        if conf.lower() == 'y':
            self.mycol.drop()

    def ShowCol(self):
        """Prints the string literal entries for tokens in the collection"""
        for document in self.mycol.find():
            print(str(document).encode('utf-8', 'ignore'))

    def is_valid(self, url):
        """
        Function from project 2, filters out traps to save time during the scrape
        """
        parsed = urlparse(url)
        if re.search(r"(\w+)(?:\W+\1){2,}/", parsed.path.lower()): #Checks if a directory in the path repeats 2 or more times (after an initial finding)
            return False
        if re.search(r"[A-Za-z0-9]{30}", parsed.query.lower()): #Checks if a the query contains a 30+ alphanumeric character string, which is usually just hash unique to a user
            return False
        if re.match("^.*month=|filter=|sort=|limit=|sessionid=|session_id=|login=|admin=|cart=.*$",parsed.query.lower()): #Checks if the query starts with month, filter, sort, limit, sessionid, or session_id; which can be infinitely long
            return False
        if re.match("^.*(/misc|/sites|/all|/themes|/modules|/profiles|/css|/field|/node|/theme){3}.*$", parsed.path.lower()): #Checks if there a 3 or more of these in the URL, which is likely in traps
            return False
        if re.match(".*\.(css|js|bmp|gif|jpe?g|ico" + "|png|tiff?|mid|mp2|mp3|mp4" \
                                    + "|wav|avi|mov|mpeg|ram|m4v|mkv|ogg|ogv|pdf" \
                                    + "|ps|eps|tex|ppt|pptx|doc|docx|xls|xlsx|names|data|dat|exe|bz2|tar|msi|bin|7z|psd|dmg|iso|epub|dll|cnf|tgz|sha1" \
                                    + "|thmx|mso|arff|rtf|jar|csv" \
                                    + "|rm|smil|wmv|swf|wma|zip|rar|gz|pdf)$", parsed.path.lower()):
            return False
        else:
            return True

    def AddPage(self, url, docid):
        """Given a url and the docid, adds a page's data to the token collection"""
        try:
            with open(url, 'rb') as html:
                soup = BeautifulSoup(html, "lxml")
                result = dict(tokenization(soup.findAll(text=True))) # Find all the text of a webpage, tokenize it, and save to a dictionary of token : count
                totalwords = len(result.keys()) # Used for tf-idf calculation
                for term,frequency in result.items():
                    try:
                        tfidf = float(frequency/totalwords)
                        self.mycol.insert_one({"_id": term, docid: tfidf})
                    except pymongo.errors.DuplicateKeyError:
                        self.mycol.update_one({"_id": term}, {'$set' : {docid : tfidf}})
        except pymongo.errors.ServerSelectionTimeoutError:
            print('ERROR : No connection could be made. Is pymongo / mongodb installed?')
            sys.exit()
        except pymongo.errors.WriteError:
            print("Skipping write (Too large to write)" + str(term)) #Strings larger than 1024 bytes are skipped - and are not necessary for queries

    def CrawlCorpus(self, directory):
        """Creates a token collection with a given directory of pages"""
        try:
            with open(directory+'bookkeeping.json','r') as inf:
                bookkeeping = eval(inf.read())
                self.pagecount.insert_one({'pagecount' : len(bookkeeping)})
                for k,v in bookkeeping.items():
                    if self.is_valid(v):
                        print("Crawling item", k, v)
                        self.AddPage(directory+str(k),k)
        except:
            raise

    def _tfidf(self, results):
        """Converts the Term Frequency of a document in the collection to tf-idf"""
        tfidf_results = {}
        for docID, tf in results.items():
            tfidf_results[docID] = tf * math.log10(self.pagecount.find_one()['pagecount']/len(results))
        return tfidf_results

    def SearchCol(self, query):
        """Searches the collection for a matching query"""
        search = tokenization(query.split())
        query_list = []
        for x in search:
            singleQuery = {"_id": x}
            results = list(corpus.mycol.find(singleQuery))[0]
            results.pop('_id', None) # Creates a list of tuples, (ID : tf)
            tfidf_results = self._tfidf(results) #Converts list to (ID : tfidf)
            query_list.append(dict(tfidf_results)) # Adds to a list a dict of (ID : tfidf)
#            sorted_list = sorted(tfidf_results.items(), key = lambda t:t[1], reverse = True) #this sorts by frequency
#            return_list.extend(sorted_list)
        return_list = [i for i in query_list[0] if all(i in d for d in query_list)] # Gets the intersection of documents for a each token in a query - Renae Lider on Stack Overflow
        return_dict = {}
        for key in return_list:
            weightedscore = float(0)
            for query in query_list:
                weightedscore += query[key]
            return_dict[key] = weightedscore
        sortlist = sorted(return_dict.items(), key = lambda t:t[1], reverse = True)
        pleasebethelastlistweneed = []
        for result in sortlist:
            pleasebethelastlistweneed.append(result[0])
        #print(len(pleasebethelastlistweneed)) #only used this for milestone1 deliverable
        return pleasebethelastlistweneed[0:19]

class Graphics:

    def __init__(self, index):
        pass



if __name__ == "__main__":
    corpus = Index('WEBPAGES_RAW/')  #This directory can be changed, as long as it is in the same directory as this .py file and has a valid bookkeeping.json file
    main_menu(corpus)
