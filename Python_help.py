'''Some stuff about WQU: 
Introduction video: in the video tap, find Youtube links taking you to the classes. 
4 one-hour office hours every week. Lecture videos will be recorded and you can access them anytime. 
The miniprojects are assigments. 
Jupyter: always duplicate the notebooks before tampering. You should get 0.9 everywhere to pass. Depending on the course, you might get bonus points. 
Note that some exercises require to pass a function. 
Style of questions available on the discussion board: how to turn a dict into tuples? 
Always include code + plain English description + tags (dict, pw, ...) 
Update the question with an answer, don't answer in the comments. Format properly.'''


#Programming and Python fundamentals 
#Conda tips
conda update 
conda install nomPackage 
conda update anaconda 
conda list 

#Jupyter tips 
Title: ## 
Keyboard functions shown in the menu 
Command mode (highlighted in blue): navigate cells with arorows, create/delete cells
Execution: ctrl + enter 
Execution + creation of new cell: shift + enter 
Shift from one mode to another: hit Escape or enter
Enter edit mode: Enter
Change cell to markdown: M 
Change cell to raw: R 
Change cell to code: Y 
Select cell above: K 
Select cell below: J 
Insert cell above: A 
Insert cell below: B, +

#Introduction to programming
'''The interpreter will only show the last result that was evaluated.
Tracebacks shows where the error was encountered. Arguments only have meanings within the function (LGI).
Be cautious about functions using the general state of the program (don't refer to variables not passed as variables
Every keyword is highlighted in green and shouldn't be modified (e.g. def) 
Documentation always includes stuff about args, calculation, returned values. 
Exceptions: every code should be able to handle exceptions - use try/except/finally'''

ExceptionType: AssertionError, IndexError (index sequence out of range), SyntaxError,
 TabError (inconsistent indentation), ValueError (function gets correct type of argument but value is incoherent)

Handling Exceptions: 
	
	assert pow(2,6) == 4, 'Error in the assertion'
	
	if condition: 
		raise TypeError, 'The word must be a string'
	
	try: 
		file = open('filename.txt')
		file.write('Python is awesome')
	except ExceptionType: 
		print('An error has occured when writing to the file')
	finally: 
		file.close()

	def check_ana(word1,word2): 
		assert type(word1) == str, 'Must be a str'
		assert type(word2) == str, 'Must be a str'
		return sorted(word1) == sorted(word2) 

Comments: '''... ''' / # 
Variables: has_dog = True, False, not True
Find vartype: type(my_variable)
Print nothing: print(None)
String interpolation: '{}'.format(my_age+10)
Variable globale: global nomVariable
Laytag: $$ f(x) = 2 $$ #Writes in proper math style
Puissance: ** / pow(variable_to_pow,power)
CheckEven: x%2 == 0
Reste de la division/modulo: 5%3 returns 2 
Update counter while: +1 
Format faster: print('First %d Fibonacci' %m) will replace %d by value of m, %s by a string
For i in liste: for i in [1,5,7]
Range: range(10) returns 0,...,9
Add counter: += 1
Input: msg = input('Want to quit?')
Lambda: sum = lambda a,b,c: a+b+c / sum(1,2,3) returns 6 #Very useful with map/lists/dicts, create custom fct in a one_liner 
Recursive functions: factorials / Fibonacci
	
	def fact(n): 
		if n == 1: 
			return n 
		else: 
			return n*fact(n-1)

Enumerate: enumerate(iterable, start) returns each iteration + counter, initialized at start. 
Quick way to multiply liste: [0]*5 = [0,0,0,0,0]
Function Closure: define a function inside a function, ex one that will raise the nb to some exponent, but we do not know which at the time
Setting Default For Args: def print(eat= True, sleep=True) Used in a complex function with many inputs, while maintaining ease of use. Allows to include a smaller nb of arguments than the ones that were defined. 
Variable Nb Args_Tuple: def print_args(*args) fait passer un tuple de longueur variable en argument 
Variable Nb Args_Dictionary: def print_args(*kwargs) où kwargs désigne le dictionnaire
Keyword Args: def pow(number = x, power = y) permet de call la fcntion en listant les args ds le désordre
Specify Arg Type: def funct(name: str)
Format floats: "%.2f" % 1.2399 returns 1.23 #See format float cheat sheet 
Capitalize string: str.capitalize()
Find position of substring: str.find('substring', index_start, index_end) #returns first index at which it is found
Find position of substring from the end: str.rfind('substring') #gives the last index 
Replace word with another: str.replace('word_replaced','replacement')
Find if any element observe a condition: any([condition]) returns True or False
Find if all elements observe a condition: all([condition])
Eval: force execution of a function stored in a variable. E.g. x = print(3+5) / eval(x) returns 8 #No need to use print
	
	All/Any: 
	N = int(input())
	lnb = list(map(lambda x: int(x), str(input()).split()))
	print(all([ele>0 for ele in lnb]) and any([str(ele) == str(ele)[::-1] for ele in lnb]))

 	Beware: 
	print(50 > 0) returns False | True and True returns True | True and False returns False |
	While requires counter update, for animal in animaux does not


#Data Structures
#Containers are heterogeneous, datasets can be represented in various dstructures (list, tuples, set, dict)
#RSI: indicateur avancé pour analyse: H = (H+B)*100 où H = moyenne mobile exponentielle des hausses au cours des n derniers jours (MMMEt = Prixt*A + MMEt-1*(1-A))
#Où B = valeurabsolue de la MME des baisses au cours des n derniers cours. Le RSI sert à repérer la puissance de un mvt. 
#MME = moyenne mobile exponentielle
# Strings are immutable (cannot be reassigned), can be sorted, are ordered, can be sliced

#Strings
String validators: 
- str.isalnum() checks if all the characters of a string are alphanumeric
- str.isalpha() checks if characters are all alphabetical
- str.isdigit() checks if characters are all digit
- str.islower() 
- str.isupper()
Split while conserving varying whitespaces: import re / re.split(r'(\s+)', line)
Split while conserving various dtypes:  [int(i) if i.isdigit() else i for i in s.split('|')]
Text wrap: to split a string into several subs of max_length: import textwrap / textwrap.wrap(text_to_wrap, max_length) returns a list with split text
Wrap and fill: to print each of the sub onto a different line: textwrap.fill(text, max_length)

	Palindromes: str(n) == str(n)[::-1]

#Listes
#Lists can be heterogeneous, are always ordered (printed the way nbs were added to the list)
# You can unpack a list (see tuples) to separate the contents. ValueError if not enough variables to unpack.
List Comprehension: [number for number in list_of-numbers if is_prime(number)] #Will add in list all the numbers verifying the condition 
Append to liste: liste.append() 
Extend to liste: liste.extend(liste2)
Remove: list.remove(object_to_remove) #Removes first instance of that object in the list, whatever the pos 
Slice: liste[0,4] / liste[3:] / liste[-100::] to count from the end of the list backwards. Returns the 100 last elements, starting with the last one. 
Get the last 3 elements; liste[-3:]
Quickly duplicate: liste*2 / [0]*5
Concatenate: liste.extend(liste_to_concat) / liste1+liste2 #Append doesn't work w/ lists
Removing items: liste.pop(indexOfItemToPop) / del liste[posItemToDel] 
Copy: liste.copy()
Spot element: if 'Google' in liste:
Reverse: liste.reverse()
Sort: liste.sort()
Clear: liste.clear()
Add element at specified position in liste: liste.insert(index, object) #insert an object before the index  
Copy liste of listes: liste.deepcopy()
Remove value from liste: liste.remove(valueToRemove) 
Count nb of values in liste: liste.count()
Display all methods available: liste.method()
Unpack a liste: a,b,c = [1,2,3] #Will return ValueError if not enough variables to unpack 
Turn object into liste: list(object_To_Transform)

#Tuples
#Tuples are immutable, but the variables contained are not necessarily of the same type. Can be accessed using index.
#Ordered (print in the same way)
#If your function returns  multiple variables,tuple is returned
#Unpacking = access the multile results of a function as separate variables 
Store multiple outputs in separate variables: return x1,x2 / rep1, rep2 = funct(x1,x2)
Return a tuple: return (x1,x2)
Unpack a tuple: name, age = (name, age)

	Beware: 
	Access tuple value: since tuples are ordered, (tuple_name)[1] 
	Cas où tuple modifiable: you can access a list that is within a tuple. 
	[Ex] tuple1 = ([liste], x) / you can write: tuple1[1].extend(liste2)

#Sets
#Sets are unorder, indicated by {set1,set2,...} - cannot use index, cannot be sliced, won't print in the same order
#Will print in a different order each time - each print is a visual representation of a set, not the set itself 
#Why are sets useful? 
Remove from set: x.remove(value_to-remove) #Removes value from the set. If doesn't exist, raises a KeyError. 
Discard from set: x.discard(value_to_remove) #Removes value from set. Does not raise a Key Error if value doesn't exist.
Pop a random objet from a set: setName.pop() #If no elements to remove, raises a KeyError. 
Add smth to a set: setName.add() / setName.update(liste) #Since append/extend doesn't work, use add/update to extend
Add a set to a set: existing_set.update(set_to_add)
Update and keep intersection with another set: existing_set.intersection_update(set_to_compare)
Update and remove elements found in another set: existing_set.difference_update(set_to_remove)
Update and keep only new elements found in another set: existing_set.symmetric_difference_update()
Apply set to other containers: set([liste]) 
Removing duplicates from list: set([liste])
Intersection sets (A and B): set1.intersection(set2)
Union sets (A or B): set1.union(set2)
Difference sets (A-B): set1.difference(set2) #set 1 - set 2
Isempty(set): bool(set_to_text) returns False
Symmetric difference sets: set1.symmetric_difference(set2) #not (A et B)

	Beware:
	When to use a set: to make data retrieval much faster through hashing 
	
	Hashing: given a set with no two identical values (wouldhave the same hash) 
	And all are immutable (otherwise, if the object were to change, its position in memory would no longer correspond with its hash)
	10 piles - Hash function = return int(x*100%31) | hash(pile) for pile in piles |
	find the 5.37 box using int(5.37*100%31) #The 31 boxes is arbitrary 
	#We find which of the 31 boxes the pile of money should be in 
	#We calculate where in memory the data should be based on the value
	Lists are unhashable; because lists might mutate and the hash wll be outdated . 
	We hash immutables: tuples, strings, numbers, booleans

#Dict
#Curly brackets, made of several key/value pairs. Generally unordered.
#Use hashing techniques on keys (/sets), cannot be accessed by index
#Advantages: to label data and quickly retrieve it without pos: are referred to as lookup tables or hashtables 
#Dict of dicts: one tag = several values as output
#In case you're going to use a complex algo several times, store results in dictionary to save computational time later
#Beware: difficult to find stuff in the dictionary based on value
Create a dictionary with zip: create key liste, value liste, zip them together

	key_list = ['name', 'age']
	value_list = ['Laetitia', '23']
	key_value_pairs_liste = list(zip(key_list, value_list)) 
	#Returns: [('name','Laetitia'), ('age', '23')]
	key_value_pairs_dict = dict(zip(key_list,value_list))

Unzip: scores, names = list(zip(*sn)) #Using the star makes the transpose
Add new key-value pair with assignment: me_dict['favorite book'] = 'The little Prince'
Add several new key-value pairs with update: me_dict.update({'fav color':'red', 'siblings':'2'})
Reassign values in dict: me_dict['favorite book'] = 'The Lion King'
Delete key and values: del me_dict['favorite book'] / me_dict.pop('siblings')
Retrieve keys: me_dict.keys()
Retrieve values: me_dict.values()
Retrieve key-value pairs: me_dict.items()
Retrieve values associated to keys: me_dict[2] returns values assoiated to key 2 
Retrieve values associated to keys with get: print('has dog: %s' % me_dict.get('has dog'))
Get with default value if key does not exist: me_dict.get('has cat', False)
Iterate over keys: for key in me_dict(): 

	Beware: 
	Keys are immutable, you cannot use a list (unhashable type)
	Since tuple are immutable, you can use tuple to create keys.
	Use tuples when you want your key to contain multiple pieces of info. 
	valid_dict = {(1,5):'a',5:[23,6]}
	Can use list for values (one key = several values returned)

# Switching dtypes, search, sorting, iteration and coprehension 
Map: map(func,seq) applies the func to all elements in a sequence #list(map(len,['Tina','Raj','Tom'])) returns [4,3,2] #Also works with custom functions
Switching datastructure: tuple(ex_list)/ set(ex_tuple) / list(ex_set)
Search in liste: print('a' in ex_list) returns True/False
Search in dictionary: print('key_1' in me_dict) returns True/False #we can search by key but not by value 
Search in dictionary values: 'brown' in me_dict returns False / 'brown' in me_dict.values() returns True
Sort tuples and sets with sorted: sorted(map(str, ex_tuple)) #Converts all elements into strings to sort easily 
Sort dictionary with sorted: sorted(me_dict.items()) / sorted(me_dict) #Returns sorted list of keys (or key-value pairs)
Sort dictionary by values: sorted(map(lambda kv: (kv[0], str(kv[1])), me_dict.items()), key = lambda kv: kv[1])
Iterating over dict items: unpack the keys and items first

	for k,v in me_dict.items(): 
		print('%s : %s' % (k,v))

List Comprehension: squares = [x**2 for x in range(10)]
Dict Comprehension: {x: x**2for x i range(10)}
Print dict_values in different lines: list(dict_values()) / for el in list, print(el)
Create new dictionary with Comprehension: 

	me_dict_dtypes = {k: type(v) for k,v in me_dict.items()}
	# New dict keys = me_dict keys, values = dtype of each me_dict value

#Collections library 
#Additional datastructures built on top of Python's: namedtuple, deque, Counter, OrderedDict, defaultdict
#namedtuple: gives tags to elements of a tuple = combination tuple/dictionary. 
#Named tuples are a way to create self-documenting code with almost no memory cost
Named tuples: 

	import collections as clt
	Vector3 = namedtuple('Vector',['x','y','z'], verbose = True) #Verbose shows class definition, Vector3 is a new object (3-dimensional vector), Vector is the name and x/y/z its attributes
	vec = Vector(1,2,3)
	vec returns Vector(x=1, y =2, z= 3)
	vec.x, vec.y, vec.z returns (1,2,3)
	#Useful since still immutable, values cannot be reassigned

	>>> from collections import namedtuple
	>>> Car = namedtuple('Car','Price Mileage Colour Class')
	>>> xyz = Car(Price = 100000, Mileage = 30, Colour = 'Cyan', Class = 'Y')
	>>> print xyz
	Car(Price=100000, Mileage=30, Colour='Cyan', Class='Y')
	>>> print xyz.Class
	Y

#Deque: double-ended queue, treatable faster than big lists when dealing with first/alst elemetns. 
#Below: same results but deque is much faster
Deque:
	
	from collections import deque 
	l_ = list()
	for i in range(40000): 
		l_.insert(0,i)  

	d = deque()
	for i in range(40000):
		d.appendleft(i)

#Counter
#Counts elements in an iterable and returns a dict structure containing the count 
Counter: 

	from collections import Counter
	ele = ['a','b','a','c','b','b','d']
	c = Counter(ele)
	#returns Counter({'b': 3, 'a': 2, 'c': 1, 'd': 1})
	c['a'] #returns nb of occurences of a. If a doesn't exist, will return 0.  
	c.most_common(2) #returns 2 most common elements + nb of occurences = [('b', 3), ('a', 2)] 

Create OrderedCounter: #Returns OrderedCounter(OrderedDict(...)) type of object. To interact, use habitual c.keys() or c.values()

	from collections import Counter, OrderedDict
	class OrderedCounter(Counter, OrderedDict):
     'Counter that remembers the order elements are first seen'
     	def __repr__(self):
        	 return '%s(%r)' % (self.__class__.__name__,
                            OrderedDict(self))
     	def __reduce__(self):
        	 return self.__class__, (OrderedDict(self),)


#Default dict: set some default value that is returned when we want to access a key that doesn't exist 
#The default dict takes a default factory function e.g. return 0 whe the ke has not been seen before. 
#Works like a normal dict, but is initialized with a function ('default factory') that takes no arguments 
#And provides the default value for a non-existent key. A defaultdict will never raise a KeyError.
#E.g ice_cream = defaultdict(lambda:'Vanilla') / price ice_cream['Joe'] will return Vanilla (if Joe doesn't exist)

	from collections import defaultdict
	def count_default(x):
    	count_dict = defaultdict(int)
    	for ele in x:
        	count_dict[ele] += 1 #For that key, if you have seen it before, add one in value of new dict 
    	return count_dict
	count_default(ele)

#OrderedDict 
#Remembers the order with which each key was included in the dictionary:
#If a new entry overwrites an existing entry, the original insertion position is left unchanged
>>> ordered_dictionary = OrderedDict()


#Dataframes
#Cleansing the dset: drop the columns which do not add values, make the format uniform, dreate a column with the years (prepare for time analysis)
#Handle the missing values: when you have localized the columns which contribute the most to the analysis, remove nulls from them to avoid the inaccuracy of the predictions - ex with mean of columns
#Finding questions you want to answer with your dset: 
#Which state has the highest CO2? Group values by state, the plot these values by mean over a year 
#What yearly trend in a particular state: follow max emission, find evolution of key historical values, plot 

Import from Excel: df = pd.read_excel(input_path)
Create df from dictionary: 

	family = {nomFam: [0,0], nompers: [0,0]}
	df = pd.DataFrame(data = family)

Export to Excel: df = pd.to_excel(nomPath)
Get first and last rows: df.head(n) / df.tail(n)
Get shape: df.shape() returns nb of rows, columns
Display column names: df.columns() renvoie un array avec ts les noms de colonnes
Quick stats: df.describe() / df.info()
Display all row labels: df.index()
Delete col: del df['NomCol'] 
Create new col: df['nomNouvelleCol'] = df['col1'] - df['col2']
Dropper des colonnes: df = df.drop(df['ColToDrop'], axis=1) # Axis = 0 for a row
Extract full name from path: ntpath.basename('filepath')
Set index: df2 = df1.set_index('nomCOlonne', drop = False) # Si drop = True, l'ancien index va être supprimé 
Extract subset with names: df.loc[startRow:endRow, 'nomColonne'] / df.loc[:,'nomColonne'] / df[['nomCol1', 'nomCol2']]
Apply mean to extract: df.loc[,:'nomCol'].mean()
Extract a single cell with loc: df.loc['nomLigne','nomCol']
Extract subset with position: df.iloc[:,0:4] #Wrn; fonctionne comme range, s'arrête en n-1
Display df methods: dir(pd.DataFrame)
Display method description: help(pd.DataFrame.mean)
Add a column manually: df['nouvelleCol'] = [valeurRow1,valeurRow2,..]
Add a column with mean in each row: df['nomNouvelleCol'] = df.mean(axis=1) #Axis = 1 pour nouvelle Col
Add a column depending on logical test: df['nomNouvelleCol'] = df.apply(x series: 'Big' if series['col1']>series['col2' else 'Small'], axis=1)
Display width: pd.display.options.width
Display max nb of rows displayed: pd.display.options.max_rows
Display max nb of columns displayed: pd.display.options.max_columns
Column dtypes: df.dtypes
Methods on aggregate: min(), max(), sum(), cumsum()
Drop NA: df.dropna(axis=0)
Fill NA: df.fillna(value_replacing_na) 
Replace NA with mean:df.fillna(mean(df.shift(-1),df.shift(+1))) / ffill / bfill
Count nb nulls in df: df.isnull().sum()
Transpose: df.T / df.transpose()
Count nb nulls in rows: df.transpose.isnull().sum() #isnull fonctionne mieux par colonne
Create categories: 

	bins = [0,10,20,40]
	names = [small, middle, big]
	df['Column_bins'] = pd.cut(df.columns_names,bins,labels = names)

Shape: df.shape #returns rows and columns
Nb of null values in each column: df.isnull().sum() 
Nb of values: df.count()
Select columns: df.columnName / df[['ColumnName']] / df.iloc[:,'columnNb']
Replace NaNs: df.fillna(newValue)
Select rows depending on logical test: df[df['Area'] == 'Ireland']
Create indicator: df['Indicator'] = np.where(df['Colonne'] > df['Colonne2'], 'Col1>Col2', 'Condition pas vérifiée')

	Alternative:
	condition1 = df['nationality'] == 'USA'
	condition2 = df['age'] > 50
	df[american & elderly] 

	Select the cases where the first name is not missing and nationality is USA
	df[df['firstname'].notnull() & df['nationality'] == 'USA']

	Rename columns df.rename(columns={'nouveauNom':'ancienNom', 
	'Y001':'Year_2001'}, inplace = True)

Rename all columns using a function: df.rename(columns =str.lower) / df.rename(columns =lambda x: x.lower().replace(' ','_'))
Merge: merged = pd.merge(user_usage, user_device([['colName1','colName2']]), on='use_id', how='joinType') #joinType = left, right, inner, outer
Factcheck the results: df1['ColID'].isin(df1['ColID']).value_counts()
Merge indicator: optionality in merge, we can ask pd to add anothercolumn after the merge indicating from where each value comes 
Skew: df.skew([axis, skipna, level, numeric_only]) returns unbiased skew normalized by n-1
Sort values df: df.sort_values(by, [axis, ascending, inplace, ...]) #df.sort_values(ascending = True) or df.sort_alues(by = ['Name Columns'], axis = 0, ascending = False)
Stdev: df.std([axis, skipna])
Convert df to dictionnary: df.to_dict([orient,info])
Convert df to csv: df.to_csv([path,sep,na_rep,...])
Display subtotals each group: groupby('nomColGroup').agg({'nomCol1ToMean':'mean', 'nomCol2ToCount':'count'})  #Où dernière instruction = on compte, pr chaque groupe, cb d'occurences ds la colonne ID
Correlation coefficient: np.corrcoef(x,y) #où x et y sont un array 
Pivot Table: pd.pivot_table(df, index=['ColumnName'], values =['colValues'], aggfunc = [np.mean])

	Warnings:
	np.array().pop() removes an element by index
	np.array();remove() removes an element by value

	Three methods to groupby:
	df.groupby('day').agg({'VIX':'mean'})
	df.groupby('day').VIX.mean()
	pd.pivot_table(df, index = ['day'], values=['VIX'], aggfunc = [np.mean])

	Stack: 
	a = np.array((1,2,3))
	b = np.array((2,3,4))
	np.column_stack((a,b))
	array([[1,2],
		[2,3],
		[3,4]])

Datetime: import datetime as dt 

	df['Date'] = pd.to_datetime(df['Date']) #we tranform to dates
	df['Year'] = df['Date'].dt.year
	df['Month'] = df['Date'].dt.strftime('%m')
	df['Monthday'] = df['Date'].dt.stdrtime('%d')
	df['Day'] = df['Date'].dt.Day
	df['Debut Mois'] = df['Date'].dt.day(<0).fillna(0).astype('unit8')
	df['Fin Mois'] = df['Date'].dt.days(>0).filna(0).astype('unit8')

'''Algorithms 
3 majors bottlenecks in the code: 
- Computational complexity: how many instructions are executed? 
- Memory needs: how much memory is needed? 
- I/O: how many reads and writes or network requests do I need to make?
The algorithm grows depending on the size of the problem. It's the algorithmic solution that has complexity. 
If we have N operations needed, we write O(N) to quantify complexity (order N). N measures the nb of inputs needed to compute the result. 
Note that only care about the dominant function of N in the expansion (hence O(N) = O(N+1) = O(2N))
If we have nested for loops = O(N^how_many_loops)
How to optimize storage? For instance, how to find a value in a list? Sorting itself = Nlog(N) complex
ity, so the more optimized the better. 
- Basic approach: run through a list (if i == ele, return True)
- Advanced1: early stop in sorted list: once you've exceeded the element you are looking for, exit the loop 
- Advanced2: binary search with recursive algorithm: allows the list to be divided roughly in half on each recursiv step, yielding a logarithmic asymptotic run time.
Steps: look for midpoint element in list. If seeked < midpoint, then discard the upper part of the list. Then repeat. Nb of steps = log(N) in base 2.
Memory complexity: if we have a list = O(N) - then list of list = O(N^2) etc
'''

Get random numbers: import random / random.randint(0, 10*N) 
Computing avg time: 
def compute(n_avgs, func, N): #Compute the avg
    times = [] #Runs the function several time and stores the timing 
    for _ in range(n_avgs):
        ts = time.time()
        func(N)
        times.append(time.time() - ts)
    return sum(times)/float(len(times)) * 1000 # milliseconds

Binary research script using recursive functions: 
def find_ele_binary(l_, ele): #l is our list, #ele the elment to find
    if len(l_) < 1:
        return False
    mid_point = len(l_)//2
    if l_[mid_point] == ele:
        return True
    elif l_[mid_point] > ele:
        return find_ele_binary(l_[:mid_point], ele)
    else:
        return find_ele_binary(l_[mid_point+1:], ele)

Memoization: to avoid repeating unnecessary steps or repeating calculation 
Steps: store results in dictionnary / inside the loop: check if result already stored / if not perform calculation 
Limits: we save computation but use more memory = its a tradeoff betwee the two

	from collections import defaultdict
	def fibonacci_mem(n, d): #where d = defaultdict
    	if n in d:
        	return d[n]
    	elif n == 0:
        	ans = 0
    	elif n == 1:
        	ans = 1
    	else:
        	ans = fibonacci_mem(n-1, d) + fibonacci_mem(n-2, d)
    	d[n] = ans
    	return ans
	fibonacci_mem(33, {0:0,1:1})

Factorial: math.factorial(n)
Enumerate a string: enumerate(string) #Returns enumerate object, to turn into list
Use enumerate to find several minimums: 

	def locate_min(a):
    	smallest = min(a)
    	return smallest, [index for index, element in enumerate(a) 
                      if smallest == element] #Means that unpacks the enumerate and only return the index in index, element

When there is more than one recursive call, use memoization to speed up the process 
If there is only one (factorial case): use memoization in a lookup table
Insert while keeping list sorted with bisect module: 

	#Python 3 program to insert an element into a sorted list
	import bisect 
	def insert(list, n): 
		bisect.insort(list,n)
		return list

Choosing your data structure wisely:
- Look at heaps (for binary search), depth (how to search in trees, networks), stacks, queus
- Call stacks: certain algorithms are often linked to particular data structures = 
- The type of datastructure to use is often dependent on the algorithm we are going to implement. 
E.g. heap (heapq library) = tree-like structure useful for order statistics, such as keeping track of the largest/smallest N items in a ocllection. 
Even as you work through your miniprojects, choosing a dictionary instead of a list will make a difference between minutes/seconds of computation. 

#Object-oriented programming (see OOB)
#Why create a class: for making plots andgraphs, creating analyzing tables of data, doing statistics and regressions
#When to define a new class: if we want to perform a set of related tasks, repeatedly
#Most libraries introduce new classes to Python (e.g DataFrame in Pandas)

- Object: anything with methods and attributes
- Variable: object that lives in memory
- Attributes: features of the state of a variable (e.g. being immutable). 
- Methods: attributes made of functions, capable of changing the state of the object (e.g. list.append())
- We cannot reassign the attributes of an object easily (e.g. x.real = 5)
- Class: blueprint for how objects should behave. Use majuscules.
- Self: always first argument of the method __init__
- Dunder methods: __init__ / __repr__ #Repr is called everytime you evaluate an object 

Find attributes and methods of an object: dir(object)
Distinguish attributes from methods: object.method() / object.attribute #No brackets
Find the class of an object: isinstance(object, expected_type)
Find the class the object inherits from: isintance(object, expected_inherited_class) #type(s) == Square / isistance(s,Rectangle) both returns True
Upper/lower case: x.upper() / x.lower()
Show documentation: x.to_bytes? #Bwr: remove method's brackets + ?
Definition of a class: 

	class Rational(object): #Definition of a class: with majuscules (LinearRegression / linear_regression() for a function)

    	def __init__(self, numerator, denominator): #Initialisation: creating the object
        	self.numerator = numerator #Create some attributes and initialize them
        	self.denominator = denominator

    	def __repr__(self): #Representation: define how it is printed
        	return '%d/%d' % (self.numerator, self.denominator)

    #Just that is enough to print(Rational(4,3)) = 4/3
    #If you get a __main__.Rational object at ... means thats you haven't defined __repr__
    #You can still access the numerator and denominator, but no meaningful rpz of the object to print out 
    #Dunder methods not meant to be called directly from the main
    #You can create supplementaty methods (besides init/repr). Note _additionalmethod(self) if not meant to be called directly (e.g. _gcd)
    #Then you can call the supplementary method in another function: gcd = self._gcd()
    

Difference private/public methods: all methods in Python are public. But _additionalmethod() would be private and additionalNormalMethod() would be public
Public methods: methods exposed to other jobjects or user interactin
Private methods: used internally to the object, often in a helper sense 
Call an instance of a class: myclass = MyClass(2) and myclass.do_it() / MyClass.do_it(myclass) #in 2nd case, you need the argument 
The problem with new classes: if you want to do math with the result of your Rational, you need to specify the basic math relations in the dunder methods __add__, __div__, __mul__ etc.
__add__ : programmer specified the operations following + #Note that even if it makes sense to you to overload it, it will be confusing to people reading the code + defining an 'add()' method is clearer 

	#A class should handle all the ways the user could want to use an object
	Defining the methods of a class:
	
	class Rational(object): 

		def reduce(self):
        	gcd = self._gcd()
        	self.numerator = self.numerator / gcd
        	self.denominator = self.denominator / gcd
        	return self

		def __mul__(self, number): #self = Rational, nb = the nb we're multiplicating against. Only works when Rational are on the left of the multiplication 
	        if isinstance(number, int): #if the number is an int
	            return Rational(self.numerator * number, self.denominator)
	        elif isinstance(number, Rational): #if it is also a Rational
	            return Rational(self.numerator * number.numerator, self.denominator * number.denominator)
	        else: 
	            raise TypeError('Expected number to be int or Rational. Got %s' % type(number))

	    def __rmul__(self, number): #To make sure that the multiplication works when Rational is on the right 
	    	return self.__mul__(number)

#Inheritance: defining subclasses of objects with particular attributes/methods. E.g. square as a subclass of rectangle. 
Inheritance:

	class Rectangle(object): #deifne class Rectangle 
    	def __init__(self, height, length):
        	self.height = height
        	self.length = length
    
    	def area(self):
        	return self.height * self.length
    
    	def perimeter(self):
        	return 2 * (self.height + self.length)

    class Square(Rectangle): #SQUARE is a subclass of RECTANGLE
    	def __init__(self, length): #Only one add argument since length = height
        	super(Square, self).__init__(length, length) #Overriding the Rectangle class with length = height # 
        	Rectangle.__init__(length, length) #Same way of writing the line above 

        def area(self): 
        	print('Calculating area of some square...')
        	return super(Square, self).area()

  	s = Square(5) 
  	s.area(), s.perimeter() #will return area and perimeter, without us having had to define the methods before


#Visualizing Data 
Matplotlib, Pandas  Visualisation, Seaborn, Ggsplot, Plotly 

	plt.plot(N_range, time_sum, 'o-', label='Sum Numbers')
	plt.plot(N_range, time_gauss, 'o-', label='Gauss')
	plt.xlabel('N')
	plt.ylabel('Average time (ms)')
	plt.legend()

	Scatterplot 1 feature
	fig, ax = plt.subplots()
	ax.scatter(df['col1'], df['col2'])
	ax.set_title('Dataset')
	ax.set_xlabel('Sepal_length')
	ax.set_ylabel('Sepal_width')

	Scatterplot several features
	colors = {'Iris':'r', 'Violet':'g','Other': 'b'} #Create a dictionary associating a dot color with afeature
	fig, ax = plt.subplots()
	for i in range(len(iris['sepal_length'])):
		ax.scatter(iris['sepal_lengthh'][i], iris['sepal_width'][i], color = colors[iris['class'][i]])
	#then set labels and columns 	