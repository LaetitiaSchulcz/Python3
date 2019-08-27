'''Some stuff about WQU: 
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
Command mode (highlighted in blue): navigate cells with arrows, create/delete cells
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
Import the zen of Python: import this 
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
Insert apostrophe in string: \t 
Variables: has_dog = True, False, not True
Round float: round(nb, ndigits)
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
Get division and rest in tuple: divmod(numerator, denominator) #Returns (num//dem, num%dem)
Lambda: sum = lambda a,b,c: a+b+c / sum(1,2,3) returns 6 #Very useful with map/lists/dicts, create custom fct in a one_liner 
Recursive functions: factorials / Fibonacci
	
	def fact(n): 
		if n == 1: 
			return n 
		else: 
			return n*fact(n-1)

	Turn input str function into actionable function 
	#Input is x,k then x**3+x**2+x+1 - and we need to test if str_funct(x) == k
	xn,k = list(map(int, input().split()))
	cmd = input().replace('x',str(xn))
	print(eval(cmd)==k) #using eval forces the exec of the str function

Enumerate: enumerate(iterable, start) returns each iteration + counter, initialized at start. 
Quick way to multiply liste: [0]*5 = [0,0,0,0,0]
Function Closure: define a function inside a function, ex one that will raise the nb to some exponent, but we do not know which at the time
Setting Default For Args: def print(eat= True, sleep=True) Used in a complex function with many inputs, while maintaining ease of use. Allows to include a smaller nb of arguments than the ones that were defined. 
Variable Nb Args_Tuple: def print_args(*args) fait passer un tuple de longueur variable en argument 
Variable Nb Args_Dictionary: def print_args(**kwargs) où kwargs désigne le dictionnaire
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
	lnb = list(map(lambda x: int(x), str(input()).split())) | [int(x) for x in liste]
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
Align string left: str.ljust(width,('-')) #fills right side with - over defined width 
Align string center: str.center(width,'-')
Align string ight: str.rjust(width,'-')
Find if str start with smth: str.startswith(letter) 

	Palindromes: str(n) == str(n)[::-1]

#Lists
#Lists can be heterogeneous, are always ordered (printed the way nbs were added to the list)
# You can unpack a list (see tuples) to separate the contents. ValueError if not enough variables to unpack.
# You can't add a scalar to a list without loop - use numpy instead 
Is empty list (ptnc): (not liste_to_check) returns True
List Comprehension: [number for number in list_of-numbers if is_prime(number)] #Will add in list all the numbers verifying the condition 
Append to liste: liste.append() 
Concatenate list: liste.extend(liste2)/  liste1+liste2 #Append doesn't work w/ lists
Remove: list.remove(object_to_remove) #Removes first instance of that object in the list, whatever the pos 
Slice: liste[0,4] / liste[3:] / liste[-100::] to count from the end of the list backwards. Returns the 100 last elements, starting with the last one. 
Get the last 3 elements; liste[-3:]
Quickly duplicate: liste*2 / [0]*5
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
Zip nested lists: list(zip(*nested_lists)) #where nested_lists = list of lists that you want to zip

#Tuples
#Tuples are immutable, but the variables contained are not necessarily of the same type. Can be accessed using index.
#Ordered (print in the same way)
#If your function returns  multiple variables,tuple is returned
#Unpacking = access the multile results of a function as separate variables 
Unpack a tuple: return x1,x2 / rep1, rep2 = funct(x1,x2)
Return a tuple: return (x1,x2) / return x1, x2

	Beware: 
	Access tuple value: since tuples are ordered, (tuple_name)[1] 
	Cas où tuple modifiable: you can access a list that is within a tuple. 
	[Ex] tuple1 = ([liste], x) / you can write: tuple1[1].extend(liste2)

#Sets
#Sets are unorder, indicated by {set1,set2,...} - cannot use index, cannot be sliced, won't print in the same order
#Each print is a visual representation of a set, not the set itself 
#Strict superset: at least has another element that does not exist in the subset 
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
Is set empty: bool(set_to_text) returns False
Symmetric difference sets: set1.symmetric_difference(set2) #not (A et B)


	When to use a set: to make data retrieval much faster through hashing 
	
	Hashing: given a set with no two identical values (would have the same hash) 
	And all are immutable (otherwise, if the object were to change, its position in memory would no longer correspond with its hash)
	10 piles - Hash function = return int(x*100%31) | hash(pile) for pile in piles |
	find the 5.37 box using int(5.37*100%31) #The 31 boxes is arbitrary 
	#We find which of the 31 boxes the pile of money should be in 
	#We calculate where in memory the data should be based on the value
	Lists are unhashable; because lists might mutate and the hash wll be outdated . 
	We hash immutables: tuples, strings, numbers, booleans

#Dicts
#Curly brackets, made of several key/value pairs. Generally unordered.
#Use hashing techniques on keys (/sets), cannot be accessed by index
#Advantages: to label data and quickly retrieve it without pos: are referred to as lookup tables or hashtables 
#Dict of dicts: one tag = several values as output
#In case you're going to use a complex algo several times, store results in dictionary to save computational time later
#Beware: difficult to find stuff in the dictionary based on value
Create a dictionary with zip: create key liste, value liste, zip them together 
#Note about zip: if argument sequences of unequal lengths, the return list is truncated to the length of the shorted arg sequence

	key_list = ['name', 'age']
	value_list = ['Laetitia', '23']
	key_value_pairs_liste = list(zip(key_list, value_list)) 
	#Returns: [('name','Laetitia'), ('age', '23')]
	key_value_pairs_dict = dict(zip(key_list,value_list))

Unzip dict: scores, names = list(zip(*sn)) #Using the star makes the transpose
Add new key-value pair with assignment: me_dict['favorite book'] = 'The little Prince'
Add several new key-value pairs with update: me_dict.update({'fav color':'red', 'siblings':'2'})
Reassign values in dict: me_dict['favorite book'] = 'The Lion King'
Delete key and values: del me_dict['favorite book'] / me_dict.pop('siblings')
Retrieve keys: me_dict.keys()
Retrieve values: me_dict.values()
Retrieve key-value pairs: me_dict.items()
Retrieve values associated to keys: me_dict[2] returns values associated to key 2 
Retrieve values associated to keys with get: print('has dog: %s' % me_dict.get('has dog'))
Get with default value if key does not exist: me_dict.get('has cat', False)
Iterate over keys: for key in me_dict(): 

	Beware: 
	Keys are immutable, you cannot use a list (unhashable type)
	Since tuple are immutable, you can use tuple to create keys.
	Use tuples when you want your key to contain multiple pieces of info. 
	valid_dict = {(1,5):'a',5:[23,6]}
	Can use list for values (one key = several values returned)

# Switching dtypes, search, sorting, iteration and comprehension 
Map: map(func,seq) applies the func to all elements in a sequence #list(map(len,['Tina','Raj','Tom'])) returns [4,3,2] #Also works with custom functions
Switching datastructure: tuple(ex_list)/ set(ex_tuple) / list(ex_set)
Search in liste: print('a' in ex_list) returns True/False
Search in dictionary: 'key_1' in me_dict returns True/False #we can search by key but not by value 
Search in dictionary values: 'brown' in me_dict.values() 
Sort tuples and sets with sorted: sorted(map(str, ex_tuple)) #Converts all elements into strings to sort easily 
Sort dictionary with sorted: sorted(me_dict.items()) / sorted(me_dict) #Returns sorted list of keys (or key-value pairs)
Sort dictionary by values: sorted(map(lambda kv: (kv[0], str(kv[1])), me_dict.items()), key = lambda kv: kv[1]) #key goes with sorted /by value
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
	>>> Car = namedtuple('Car','Price Mileage Colour Class') #Car is a 4 dimension object/tuple? 
	>>> xyz = Car(Price = 100000, Mileage = 30, Colour = 'Cyan', Class = 'Y')
	>>> print xyz
	Car(Price=100000, Mileage=30, Colour='Cyan', Class='Y')
	>>> print xyz.Class #Returns Y

#Deques
#Double-ended queue, treatable faster than big lists when dealing with first/alst elements
#Below: same results but deque is much faster
#Works with append, clear, extend, remove, reverse
Append left: d.appendleft(number)
Append right: d.appendright(number)
Extend left: d.extendleft('234') # Add 2, 3, 4 as list
Extend right: d.extend('234')
Shift queue: d.rotate(3) #[1,2,3] becomes [3,1,2]

	
	from collections import deque 
	l_ = list()
	for i in range(40000): 
		l_.insert(0,i) #insert i at pos 0 = list will start with 4000  

	d = deque() 
	for i in range(40000):
		d.appendleft(i)

#Counters
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

#DefaultDicts
#set some default value that is returned when we want to access a key that doesn't exist 
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

#OrderedDicts 
#Remembers the order with which each key was included in the dictionary:
#If a new entry overwrites an existing entry, the original insertion position is left unchanged
Create orderedDict: ordered_dictionary = OrderedDict() 

#Itertools
Cartesian products: itertools.product(lists_or_tuples) / itertools.product(liste, repeat = nb) #Produit cartesien de la même liste/du même tuple
Permutations iterable: itertools.permutations(iterable [,r]) #Returns successive rlength permutations of  elements in an iterable. 
#If r is not specified or is None, r defaults to length of the iterable 
#And all possible full length permutations are generated. Permutations are printed in a lexicographic sorted order
#If the input iterable is sorted, the permutation tuples will be produced in a sorted order 
Combinations iterable: itertools.combinations(iterable[, r]) 
Combinations with replacements: itertools.combinations_with_replacement(iterable, r) #r-length tuples in sorted order, with repeated elements 
Filtering: iffilter(filter, seq) / iffilter(lambda x: x%2, range(10)) #returns 1 3 5 7 9 
Filtering on opposite condition: iffilterfalse(filter,seq) #same as previous returns 0, 2, 4, 6, 8 
Filter while: takewhile(filter, seq) / takewhile(lambda x: x<5, [1,4,6,4,3]) #returns 1 4 
Slice string: islice(seq, start, stop[, step]) / islice('ABCDEFG',2,None) # returns C D E F G 

#Calendar functions
#Calendar functions available here: https://docs.python.org/2/library/calendar.html#calendar.setfirstweekday
Import calendar: import calendar 
Generate plain text calendars: calendar.TextCalendar(firstweekday=6) #0 is Monday, 6 is Sunday 
Set first week day: calendar.setfirstweekday(calendar.SUNDAY) #Could be MONDAY, TUESDAY, WEBDNESDAY
Get current first wek day: calendar.firstweekday()
Check if leap year: calendar.isleap(year)
Get day: calendar.weekday(year, month, day) #retuns day of week - 0 is Monday for year (1970...), month(1-12), day(1-31)

#Dataframes
#Cleansing the dset: drop the columns which do not add values, make the format uniform, create a column with time (prepare for time analysis)
#Handle the missing values: when you have localized the columns which contribute the most to the analysis, remove nulls from them to avoid the inaccuracy of the predictions - ex with mean of columns
#Finding questions you want to answer with your dset: 
#Which state has the highest CO2? Group values by state, the plot these values by mean over a year 
#What yearly trend in a particular state: follow max emission, find evolution of key historical values, plot 
#Beware with pandas: sometimes the csv rows will contain less data than index (some is missing). Should be indicated by ',,' on the csv file. 
#Upon import, pandas is unable to detect the case and just places a NaN at the end of the row (no matter where the NaN should have been placed)
#Hence: lignes décallées: data values are assigned to the wrong cell
#Hence: always inspect the dstructure (checking the NaNs)
#How to fix the badly formatted CSV? File directly = add a ',' where it is missing (inefficient)
Import from Excel: df = pd.read_excel('filepath')
Print raw csv: !cat ./data/csv_name.csv
Import from csv: df = pd.read_csv('filepath', index_col = 0)
Write to to csv: pd.to_csv('filepath')
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
Sort values df: df.sort_values(by, [axis, ascending, inplace, ...]) #df.sort_values(ascending = True) or df.sort_values(by = ['Name Columns'], axis = 0, ascending = False)
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
Show current date: datetime.datetime.now().date()
Show time delta: datetime.timedelta(_timeofyourchoice_)

	df['Date'] = pd.to_datetime(df['Date']) #we tranform to dates
	df['Year'] = df['Date'].dt.year
	df['Month'] = df['Date'].dt.strftime('%m')
	df['Monthday'] = df['Date'].dt.stdrtime('%d')
	df['Day'] = df['Date'].dt.Day
	df['Debut Mois'] = df['Date'].dt.day(<0).fillna(0).astype('unit8')
	df['Fin Mois'] = df['Date'].dt.days(>0).filna(0).astype('unit8')

#Algorithms 
''' 3 majors bottlenecks in the code: 
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
Split string into letters: enumerate(string) #Returns enumerate object, to turn into list
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

#Object-oriented programming (OOB)
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

#Inheritance
#Defining subclasses of objects with particular attributes/methods. E.g. square as a subclass of rectangle. 

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

#BASH commands 
#More on: https://medium.com/cameron-nokes/bash-commands-ive-found-most-useful-for-front-end-development-df66c8544c96
Quick pick inside file: !cat filepath.extension
Summary of file: !less server.log
Open file with default app: open index.json 
Open file with custom app: open index.json -a TextEdit
Open current directory: open . 
Create directory: mkdir my-folder
Create subdirectory and parents: mkdir -p level1/level2/level3/level4 #Without the -p, mkdir will fail on non-existent directories
Delete directory (all files, subs included): rm -rf level1 # -r for 'recursive', '-f' for force
Create empty file: touch new_file.js / echo > myfile.js
Create, initialize file: echo "console.log('hello')" > index.js
Find with name: find . -name "part_of_filename*.js" #Accepts wildcards but not full regex
Fihttps://www.programiz.com/python-programming/examples/conversion-binary-octal-hexadecimalnd, delete: find . -name "*.log" -delete
Download file: wget http://.../.json.gz -nc -P ./pw-data #wget + full file path + receiving directory 

#Files
#In scala: Dstructures immutables; functions declare what type of args they accept and the type they return 
#Readability comes first: simple > complex, flat > nested, sparse > complex 
#No multiple brackets as in Java, use minimalistic notations, one liners better than long text. Make it elegant.
#Errors shouldn't pass silently! Error messages are a tool, so use the handlers. 
#In the face of ambuity, refuse the temptation to guess. Ambiguity whenever there is complexity. 
#Keep the code explicit with as many comments as necessary.
#Explicitely silencing an error: with try:except. But precise the exception you want to catch. 
#Try/Except: aka 'ask for forgiveness, not permission'. Should remain exceptional.
#Namespaces? 
#Flags: r/w/a/r+ (read + write)
#We cannot open a file for reading if it doesn't already exist; but we can open it for writing (a new file)
#Types of files: cvs (data splitted by comas), json, gzip, pickle 
Show file (fast): !cat ./filepath.extension 
Compare length of files: !ls -lh ./filename* #* to consider several formats of the same file
Open file: f=open('./data/sample.txt','r') #Where f = file handler | 'r' = flag/mode, can be 'r' (read) or 'w' (write) or 'a' (append) = adds to the file, but doesn't override what was written before
Read file: data = f.read() #File Handle = IO wrapper, with encoding etc
Read one line: f.readline() / #using it twice will skip to the next line, as input() in HR
Read all lines: f.readlines() #List of strings, each string = one line
Write on file: f.write('text. \n')
Strip a line: line.strip() #Removes extra character in the file like \n
Close file: f.close()
Context handler: #Only opens the file inside the with, then automatically close it at the end of the with 

	with open('filepath', 'r') as f: 
		print(f.read())

	with open('filepath', 'w') as f: 
		f.write

Deleting one line from a file: #Stteps: open > read and get all lines > go write mode > write back all lines except the one you want to discard: 

	Long version:
	with open("yourfile.txt", "r") as f:
    	lines = f.readlines()
	with open("yourfile.txt", "w") as f:
    	for line in lines:
        	if line.strip("\n") != "nickname_to_delete": 
            	f.write(line)

    Single open with r+: 
    '''To consider: I wouldn't do this. If you get an error in the for loop
    you'll end up with a partially overwritten file, with duplicate lines or a line half cut off. 
    You might want to f.truncate() right after f.seek(0) instead. 
    That way if you get an error you'll just end up with an incomplete file. 
    But the real solution (if you have the disk space) is to output to a temporary file and then use os.replace() 
    or pathlib.Path(temp_filename).replace(original_filename) to swap it with the original after everything has succeeded.'''
    with open("target.txt", "r+") as f:
    	d = f.readlines()
    	f.seek(0)
    	for i in d:
        	if i != "line you want to remove...":
            	f.write(i)
    	f.truncate()

Package to navigate file system: os 
Browse through directory: import os / os.listdir('.') # '.' for curr directory, dirpath otherwise
Browse through subdirectory: os.listdir('./name_of_sub') #Subs obv do not have file extensions
Browse through the location + all subs: os.walk(top, topdown = True, onerror= None,...)
Set where to seek files: os.chdir('/home/user/directory_to_look_at')
Create directory: os.makedirs(name, mode=?, exist_ok = False)
Show paths to files using walk: 
	
	Show everything
	for root, dirs, files in os.walk('.'):  # ~ or /home/user/directory. Os.walk generates a 3-tuple (dirpath, dirnames, filenames)
    	for file in files: 
        	print(os.path.join(root, file))

	Show paths for csv only
	shwfiles = []
	for dirpath, subdirs, files in os.walk(path):
    	shwfiles.extend(os.path.join(dirpath, x) for x in files if x.endswith(".csv"))

Parsing without Pandas: #Could use namedtuples also

	list_tables = []
	with open('./data/csv_sample.txt','r') as f: 
		for line in f.readlines():
			row = lne.strip().split(',')
			list_table.append((int(row[0]), row[1], int(row[2]))
	list_table returns liste of listes (each represents a row)

Downloading file if not in directory: 

	if 'factbook.csv' not in os.listdir('./data/'): #Replace by os.walk('./data/') to explore all the subs
    	!wget -P ./data/ https://perso.telecom-paristech.fr/eagan/class/igr204/data/factbook.csv
	countries = pd.read_csv('./data/factbook.csv', delimiter=';', skiprows=[1])

	%% bash 
	mkdir pw-data
	wget http://dataincubator-wqu.s3.amazonaws.com/pwdata/practices.json.gz -nc -P ./pw-data

Get filename: f.name
Tell current position of pointer in the file: f.tell()  #in bytes
Use seek to change pointer position: f.seek(offset, whence) #Where offset is the nb of characters, whence = pos where starts reading (os.SEEK_CUR for current pos, os.SEEK_SET for beginning, os.SEEK_END to go at the end of the file). Note: moving the cursor = f.read() will display the remaining line
Find row: print('found at line %s' % [num for num, line in enumerate(f, 1) if 'data' in line])

	lookup = 'the dog barked'
	with open('filepath', 'r+') a f: 
		for num, line in enumerate(f, 1): 
			if lookup in line: 
				print('found at line', num)

#JSON
#Data received from API (twitter, google...) will be JSON (JavaScript Object Notations) 
#Data structure made of nested dictionaries and lists 
#Advantage: doesn't have to conform to tabular structure, can be shared quickly over the internet (hence must be kept small = no redundancy of data)
#Problem: typical json file structure is tedious to write - typically requires to loop through dictionaries or write repeatedly the tags
#JSON subtypes, regular JSON, new JSON, simple JSON 
#JSON files are just text files. If we just read them = will only appear as one string. 
#Warning: writing json files based on Python entries will turn tuples into lists, cannot represent a dummy class...
Import JSON library: import json 

	Typical JSON: 
	book1 = {'title': 'The Prophet',
         'author': 'Khalil Gibran',
         'genre': 'poetry',
         'tags': ['religion', 'spirituality', 'philosophy', 'Lebanon', 'Arabic', 'Middle East'],
         'book_id': '811.19',
         'copies': [{'edition_year': 1996,
                     'checkouts': 486,
                     'borrowed': False},
                    {'edition_year': 1996,
                     'checkouts': 443,
                     'borrowed': False}]
         }
         
	book2 = {'title': 'The Little Prince',
         'author': 'Antoine de Saint-Exupery',
         'genre': 'children',
         'tags': ['fantasy', 'France', 'philosophy', 'illustrated', 'fable'],
         'id': '843.912',
         'copies': [{'edition_year': 1983,
                     'checkouts': 634,
                     'borrowed': True,
                     'due_date': '2017/02/02'},
                    {'edition_year': 2015,
                     'checkouts': 41,
                     'borrowed': False}]
         }

Create JSON from python dstructure: import json / with open('./file.json', 'w') as f: / json.dump(library, f, indent = 2) #Then !cat ./file.json to show the properly formatted json 
Load JSON into dstructures: import json / with open('./file.json', 'r') as f: / reloaded_library = json.load(f) 
Parse JSON string: json.loads(f.read()) #will only work with f.read()#Will return a list instead of a string (f.read())
Pandas read JSON: pd.read_json('./file.json') #Will try reading the json into a table - but will show hierarchies in cells (not useful)
Pandas read JSON from API/Internet: pd.read_json('https://api.github.com/repos/.../issues?per_page=2')

#Gzip
#Compressed files use .zip or .gz (use deflate algo?)
#Idea: look for repeated patterns in the file and points to the last appearance of a pattern and length of the match with that pattern
#Compressed files are binary: hence 'wb' as a flat (write binary)
#Won't compress well on random data or small files
#On binary files, the !cat method will only show filename + some other stuff = cannot read 
Import gzip library: import gzip
Open gzip: with gzip.open('./file.gz', 'wb') as f: / f.write(text.encode('utf-8')) #Encode if non ASCII characters - nothing to do with gzip

#Pickle
#Pickle is an example of serialization = often, we work with complex objects stored in memory; but we may need to move these objects to another computer
#Or write them persistently in memory so that we can shut down the interpreter and work with them again another time
#Ex: ML model that we spent a long time training = serialize it to disk to send it to a network or reuse it another time to make predictions
#Json & csv can be used to write smth in text on disk. What about more complex formats? (ML models...)
#You can pickle a dummy class
Import pickle library: import pickle
Create pickle: with open('./filename.pkl', 'wb') as f: / pickle.dump(pickle_example, f)
Load pickle: with ... as f: / loaded_ex = pickle.load(f)
Pandas read pickle: pd.read_pickles(data = '')
Pandas export to pickle: pd.to_pickle('filepath.pkl')
Pickle a dummy class: 

	with open('./dummy_class.pkl', 'wb') as f: 
		pickle.dump(DummyClass, f)

#NumPy - file formats (.npy)
#Along with SciPy = part of the older Python packages. Both were the tools that made 
# Python an attractive programming language. What does it offer: 
''' 
- Mathematical functions: sin, log, floor, ceiling, exponential... 
-  Random Submodule (useful for random sampling)
-  NumPy ndarray = n dimensional array. Similar to a n-dimensional matrix or tensor. Works like a list of list, 
where the depth of the inner list makes up for the array dimension. Each list is a row in this array. 
Not limited to 2 dimensional (matrix), 3 (cube - then you have row/columns/depth...) 
Such arrays are often used for representing images, in which case they are two-dimensional. 
Note that a numpy array has a shape =/= list of lists, where each inner list has a different length. 
Everytime we have an array with different dtypes inside, then i'll  consider the dtype as object.
If dtype('<U5') just means that the memory allocated is smaller than that. 
Inside an array: at best, boolean/int/floats. Outside of that, choose another dstructure.
#Summing the numbers in an array is much easier than for a list of lists. 
#Why is it better than lists of lists: when a list is summed, Python needs to figure out the correct way to add 
#the elements of the list together v when ndarray is summed, NumPy already knows the type of each element (and they are consistent)
#this, can sum without checking the correct add function for each element 
#List stuff: every time Python is adding up nb, it wonders what addition means. Can add 6 + True considering True = 1. 
#NumPy is 10x faster + has a much better syntax for performing computations on collections of numbers
#NumPy is the basis for many other datascience packages
'''
Import numpy: import numpy as np 
Turning a list of lists into a numpy array: array1 = np.array(list_of_lists) #If the inner lists are of different length, then it'll consider each row = 1 list; and make it a one dimensional object. 
Numpy export array to text: np.savetext('filepath.txt', sample_array)
Sum number inside array: an_array.sum()
Sum of nb inside array rows: an_array.sum(axis = 1)
Sum of nb inside array columns: an_array.sum(axis=0)
Add scalar to array: an_array + 5 #returns each value in the array + 5 
Multiply array by scalar: array*5
Divide array by scalar: array/5
Add two arrays: array1 + array2 
Sum inside array: np.sum(my_array, axis = 0/1/None) #If None, returns sum of all values
Prod inside array: np.prod(my_array, axis = 0/1/None)
Transpose array: np.transpose(array) 
Inner/outer matrix product: np.inner(A,B) / np.outer(A,B)
Add arrays: np.add(a,b)
Substract arrays: np.substract(a,b)
Multiply arrays: np.multiply(a,b)
Divide arrays to float: np.divide(a,b)
Divide arrays to int: a%b / np.mod(a,b)
Power arrays: a**b / np.power(a,b) #returns a^b
Find coef polynomial given the roots: np.poly([list of roots]) 
Find roots polynomial given the coefficients: np.roots([list coefs])
Find antiderivative (indefinite integral) of a polynomial: np.polyint([list coefs])
Find derivative of a specified order of a polynomial: np.polyder([list coefs], m= k) #where k = order of differentiation
Evaluate polynomial at specific value: np.polyval([coefs], value) 
Find polynomial of specified order fitting set of data (LS approach): np.polyfit([list data1], [list data2], order) #returns coefs
#See polyadd, polysub, polymul, polydiv to handle basic arithmetic for polynomials
Cross product of two arrays: np.cross(arr1, arr2) #returns prod diag - prod opposite diag 
Turn array into flat line: array.flatten()
Mean of array rows: np.mean(array, axis =0) #compute mean for each row
Mean of array: np.mean(array) / np.mean(array, axis = None) #Works the same with np.var() and np.std()
Multiply two arrays (element wise): array1 * array2
Divide two arrays (element wise): a/b.astype(float) #Note that the dividing array or scalar must be a float 
Dot product of two arrays: np.dot(a,b) #inner product - multiply all possible pairs, create a n*n matrix as the output (from a 1*5 * 1*5) 
Outer product of two arrays: np.outer(a,b)
Numpy export array to npy: np.save('filepath.npy', sample_array)
Numpy load array: np.load('filepath.npy')
Get array shape: an_array.shape 
Get array dtype: an_array.dtype #np arrays have homoegeneous dtypes:each element could be an integer with 64 bits of memory attached (int64) 
Create 1dim array with start, end, nb of dpoints equidistant: np.linspace(0,10,5) #creates 5 dots between 0 and 5 separated by same distance #BEWARE: LINSPACE KEEPS THE UPPER BOUND
Create 1dim array with start, end, stepsize: np.arange(1,10,1)
Create 1dim array with logarithmic-spaced points: np.logspace(1,10,10) #Goes from 1^10 to 10^10. By default, base is 10. 
Create 1dim array with zeros: np.zeros(10) #default type is float, can change using np.zeros((nb rows,nb cols, depth), dtype = np.int)
Create 3dim array with zeros: np.zeros(depth, height, length) # 2,10,5 will return two similar matrix 10 rows * 5 columns
Create diagonal array: np.diag([1,2,3,4]) returns a 4*4 matrix with the list in diagonal
Create identity matrix: np.identity(5) returns 5*5 matrix with ones in diagonal
Create diagonal matrix with ones: np.eye(nb rows, nb cols, k) #k > 0 = positive diagonal, k< 1 = negative, k= 0 if main diag (identity)
Change type of nb inside array: array.astype(int) #verify with array.astype(...).dtype
#Note that you can convert booleans into floats doing; np.array([True, False]).astype(float) #false = 0 | true = 1
Universal functions: NumPy defines a ufunc that allows to run functions over array. Many of these functions, like np.cos are built in and compiled with C code. 
# The function peform broadcasting, which allows them to automatically handle operations between arrays of different shapes (for ex 2 arrays wth the same shape, or one array + one scalar)
Select part of array: array[row_nb_start:row_nb_end, column_nb_start:column_nb_end]  #select from first column to ante-last = 0:-1
Apply mean to array: array[:,:].mean()
Get max in array: array.max(axis = 0/1)
Get floor in array (element-wise): np.floor(my_array) #Returns floats
Get ceiling in array (element-wise): np.ceil(my_array) #returns floats
Round to nearest integer in array (element-wise): np.rint(my_array)
Round to a certain decimal place: np.around(a, decimals = 0, out = None)
Find index at which max occured: array.argmax() #Add 1 to get actual nb, since count starts at python
Turn array into list: array.ravel().tolist() #Returns one big column with all the values in order
Slice array according to condition: random_array[random_array % 2 == 0] returns an array with only even elements
Conditional filtering: random_array % 2 == 0 #Returns array of the same size as original, with only boolean (True or False depending on the condition)
Extract elements from array according to index: array[[index1,index2,index3,index4]
Extract value from array according to matrix pos: array[1,1] #returns value at pos 1,1
Reshape array; array.reshape(new_nb_lignes, new_nb_columns) #Valid as long as the new matrix contains the same nb of values
#You could change shape by modifying the attributes directly 
Horizontal Stack: np.hstack(a,b) #turns everything in a line - concatenate
Vertical stack: np.vstack(a,b) # one row below another
Depth stack: np.dstack(a,b) 
Turn array into 1D array (list): matrix.ravel()
Transpose the array: matrix.transpose() #Also valid for non square matrices

	Find week periodicity with arrays: #Array representing one month = 4*7 (7 days, 4 weeks) = 7 columns, 4 rows
	from fractions import Fraction
	normalized_sales = (jan_coffee_sales - jan_coffee_sales.mean()) / abs(jan_coffee_sales - jan_coffee_sales.mean()).max()
	frequencies = [Fraction.from_float(f).limit_denominator() for f in np.fft.fftfreq(normalized_sales.size)]
	power = np.abs(np.fft.fft(normalized_sales.ravel()))**2
	list(zip(frequencies, power))[:len(power) // 2]

	#Returns frequencies 
	#Like Fraction(1,28) = once every 28 days...
	Weekly rythm = Fraction(1,7) and the number on the right points at the power of the signal for that frequency 

#Linear Algebra submodule: linalg 
#Check documentation: much more available!
Get determinant of array: np.linalg.det(my_array)
Find eigenvalues and eigenvectors of square array: values, vectors = np.linalg.eig(my_array)
Inverse of matrix: np.linalg.inv(my_array)

#Random Submodule
Random values in given shame: rand(d0, ..., dn)
Sample from std normal distrib: randn(d0,...,dn)
Random integer: randint(low, high, size = (nb rows, nb columns))
Radom floats in [0,1[ : random_sample(size)
Random sample chosen in 1D array: choice(array, [size])

#Visualizing Data with Matplotlib  
#plt.plot(smth) will draw over the same graph, if repeated, as long as the .plot are included in the same notebook cell 
#plt is a state machine: all mods you make depend on the current state of plt
#If plt is closed, calling plt gets you a new figure for your times series
#Not necessarily clear from the code itself to figure what's going to happen.
#Hence, name your figures instead = object-centered model. 
#Note that writing to create plots is a task customized to the data you are working on - don't spend time creating functions. 
#Also, don't spend too much time working on plottig libraries: instead, draw a picture of the graph you'd like to make (based on the dset you have)
#And then, go dig into seaborn/matplotlib docs = they have galleries of pictures of what people have made, and code is available. 
#Hence: copy the relevant code and adapt it. 
Matplotlib, Pandas  Visualisation, Seaborn, Ggsplot, Plotly 
Multiple figures: plt.figure() / plt.plot(smth) / plt.figure() / plt.plot(smth_else)
Importer: import Matplotlib.pyplot as plt 

	Stock price function (with arg = nb of days and initial price)
		
		def gen_stock_price(days, initial_price):
	    	# stock price grows or shrinks linearly
	    	# not exceeding 10% per year (heuristic)
	    	trend = initial_price * (np.arange(days) * .1 / 365 * np.random.rand() * np.random.choice([1, -1]) + 1)
	    	# noise will be about 2%
	    	noise = .02 * np.random.randn(len(trend)) * trend #using random from normal distrib
	    	return trend + noise

	    days = 365
		initial_prices = [80, 70, 65]
		for price in initial_prices:
	    	plt.plot(np.arange(-days, 0), gen_stock_price(days, price))
			plt.title('Stock price history for last %d days' % days)
			plt.xlabel('Time (days)')
			plt.ylabel('Price (USD)')
			lt.legend(['Company A', 'Company B', 'Company C'])


Name a figure: fig = plt.figure(figsize =(length,height))
Add gen title to subplots: fig.suptitle('Suptitle')
Create histogram: ax1.hist(data)
Create scatterplot: ax1.scatter(data_axis_x,data_axis_y, label='Title')
Set labels for scatterplot axes: ax1.set_xlabel('name of label1') / ax1.set_ylabel('name of label 2')	
Show legend: plt.legend()
Customize plot line: plt.plot(N_range, time_sum, 'o-', label='Sum Numbers')

	Create subplots: 
	ax1 = plt.subplot(221) #2x2 squares, pos = 1 starting from top left corner. 
	ax1.hist(shoes)
	ax1.set_xlabel('Shoe size')
	ax1.set_ylabel('Counts')

	Scatterplot several features
	colors = {'Iris':'r', 'Violet':'g','Other': 'b'} #Create a dictionary associating a dot color with afeature
	fig, ax = plt.subplots()
	for i in range(len(iris['sepal_length'])):
		ax.scatter(iris['sepal_lengthh'][i], iris['sepal_width'][i], color = colors[iris['class'][i]])
	#then set labels and columns 	

	#Show reg line on your scatterplot
	fit_line = linregress(shoes, jerseys)
	ax3.plot(shoes, fit_line[1] + fit_line[0] * shoes, 'r', label='Line of best fit')


#Pandas 
#Much more powerful than Excel! 
#PDs gets us to name the fields or entities of our data structure = no need for remembering the index or positioning. 
#Each column repreents a different type of data = aka "field"
#Each row is named "record" or "entity"
#How to create dfs besides importing the data: use zip, excel, csv, json?
#Panda series: a single column of data, with a name (field name) and an index (number  if index not set, or string/tuple/boolean... if index set)
#A df is essentially a dictionary of data series = access fields using df['name of field']
#Trying to retrieve an unexisting series will hence return a KeyError
#Since all series are built using np arrays, each series is expected to have a homogeneous dtype (object, bool, int, float mostly)
Describe df: df.describe()
Df shape: df.shape
Find index: df['series_name'].index
Access a cell based on name and pos: df['series_name'][100] #where 100 = row pos
Creating df and set index: df = pd.DataFrame({'shoe_size': shoes, 'jersey_size': jerseys}, index = players)
Apply numpy functions to df: np.log(df) #Only if df fields of appropriate dtype
Show field means: df.mean()
Show entity values with index: df.loc['Ronaldo']
Show multiple fields: df.loc[[field1,field2],:]
Show multiple continuous fields: df.loc['Ronaldo':'Best', :]
Position-based indexing; df.iloc[:,:]
Add a new row with index: df.loc['Dylan'] = {'jersey_size': 91, 'shoe_size': 9, 'position': 'midfield'} #note the brackets
Add a new column: df['position'] = np.random.choice(['goaltender', 'defense', 'midfield', 'attack'], size=len(df)) #Where each row value is chosen randomly among the list 
Build df from dict: df = pd.DataFrame({'name_field_1': list1, 'name_field_2':list2}) #Makes sense: built the columns as lists, then assemble them together

	Build df from np array: 
	# 1/ Generate a np array using rand 
	# 2/ Name the rows, then the index 
	random_data = np.random.random(4,3) #matrix 4 rows, 3 columns
	df_random = pd.DataFrame(random_data,columns=['a','b','c'], index=['row1','row2','row3','row4'])

#WORK WITH INDEXES - bc it's easier to locate data & if indexes bt df match, you can merge them
Drop row: df.drop(row_number) #drop(4) would work for the 5thr row + omitting inplace = True creates a copy of the original df 
Set index: df.set_index('name of field', inplace=True) #could have been done upon index creation
Reset index to be an enumeration of rows: df.reset_index(inplace=True)
Set index name: df.index.name = 'name of index' 
Set multiple indexes: df.set_index(['name field 1', 'name field 2'])
Add df with same index: df_a + df_b

#Reading data from file 
#Pd can read from Json, HTML, csv, Excel, Python, pickle or even use a database connection 
#There are plenty of options for pd.read_csv(), check documentation 'IO Tools'  
#Excel: you can handle retrocompatibility as well
Display pandas version: pd.__version__ 
Read csv and set index: pd.read_csv('filepath',index_col = pos_col) #specify a nb as pos_col
Read csv without header and add field names: pd.read_csv('filepath', names = ['...'], header=None)
Read csv and specify delimiters: pd.read_csv('filepath', delimiter= '\t')
Include datetime of row data in csv df: pd.read_csv('filepath', parse_dates = ['dates of creation']) #the new column will be of datetime dtype

	Loading json into df: #Long 
	import gzip
	import simplejson as json
	with gzip.open('./data/yelp.json.gz', 'r') as f: #decompress the file
	    yelp_data = [json.loads(line) for line in f] #parse the file into a json dstructure
	yelp_df = pd.DataFrame(yelp_data)  #turn into df 
	yelp_df.head() #Note that the table contains dstructures (dictionaries or list) = unusual but still works well in our case (the data isn't "flat")    

	Loading json into df: #Short
	#Note that the json may be stored in a file as a long string or multiple lines in the file
	#Hence in the read_json options, 'lines' arg (boolean) can be set to True (default: False)
	#Most important keyword = 'orient': tell pandas if the json is a list of dictionaries (orient: records), less common = split (see doc)
	#Hence: 1/ Check format of your json using: !zcat filepath.gz | head
	#If records:many dictionaries, within which each key is a column name.
	# 2/ set orient 
	# 3/ Specify if compressed
	df = pd.read_json('filepath', orient= 'records', compression = 'gzip')

	Unzipping gzip csv into pd: 
	df = pd.read_csd('filepath.csv.gz', compression = 'gzip')


#Filtering 
#Note that the filter can come from another df, as long as the two dfs share the same index or they share a column
#Logical operators: ~ = Not, | = Or, & = And
#Contains is useful when names are not properly formatted
Filter df based on conditions: df[(df['state'] == 'Arizona') and (df['review count'] >10)] #writing df['state'] == 'AZ' returns series of booleans (True or False for each row)
Count how many results are filtered: df[df['state'] == 'Arizona'].count() / df['city'].nunique() #nb of distinct entities
Use negation to filter: df[~df['open']] #returns all the businesses that are closed
Verify df filter: df['state'].unique() #Returns an array of the same dtype, showing the condition you filtered on 
Filter on contain string: pd.Series([....]).str.contains('word you want') or df[df['city'].str.contains('Vegas')] #first option returns a series of booleans

#Apply 
#Note that transforming the data using stat functions only works with relevant dtypes. If one cell contains a dict or a list, much harder to deal with.
Apply log: new_col = np.log(df['columns to log'])
Apply mean: mean_of_col = df['review_count'].mean()
Show all means: df.mean() #will only return means for relevant dtypes (float, int)
Show column names: df.columns

	Interact with cells containing a dict:  
	def get_delivery_attr(attr_dict): 
		return attr_dict.get('delivery') #returns true or false depending on the value of delivery for each dict in the attr column
	
	# For one shot check, use: get_delivery_attr(df.loc[0,'attributes']) #interact with first dict 
	delivery_attrib = df['attributes'].apply(get_delivery_attr) #Apply that function to each row. Note that we don't call the function = no parenthesis, no args

Fill NA with False: df['delivery'].fillna(False).head() #not a mutation since we don't use inplace = True
Find all restaurants with delivery: df[df['categories'].apply(is_restaurant)]['delivery'].fillna(False).mean() #fill NA is not optional 
Use lambda functions (to use a function once, then erase it): df[df['categories'].apply(lambda x: 'Restaurants' in categories)]['delivery'].fillna(False).mean() #lambda replaces def to use a function in a one shot

#Aggregate with groupby
#Aggregation = overloaded term, might refer to summarizing data (ex: computing mean) or actually aggregatin gdata 
#Pattern of work: split - apply - combine => See documentation on pandas
#Splitting: th data into some groups
#Apply: some function to each group = represent each group by a subset or some statistics
#Combine the results into a final datastructure
#Whenever we group by smth, the distinct entities we chose to group by become the indexes of our grouped df 
#The groupby method returns a "groupby object" (can be a series.groupby.object or df.groupby.object)
#Loads of aggregation boil down to min/max/count/sum etc
#We can build a groupby object then select a series from a groupby object.  
#Tips for transformation and aggregation 
#NO ITERATION RULE: If ever you start writing 'for smth in pandas_object': there's always a better way of doing it by using this pandas_object and apply a function: pandas_object.apply() 
#Hence: never explicitely iterate over pandas object, apply or aggregate instead
#Apply or aggregate don't need arguments. We always need to think before applying: 'what would this argument of this function should be?' Should be a group df or a series.
#ARGUMENT RULE: always think of the argument you would pass to the function.
Group df by smth: df_grouped = old_df.groupby('city')
Display groups: df_by_city = df.groupby('city') / by_city.groups #Returns a dictionarry with the cities as keys and indices of all the values belonging to this group as dict values
Extract get stars mean by city: stars_by_city = df.groupby('city')['stars'].mean()
Show methods aplicable to the groupby object: dir(grouped_df)
Show mean for each group: grouped_df.mean()
Retrieve particular group: grouped_df.get_group('Anthem') / df[df['city'] == 'Anthem']
Aggregate and specify methods: agg_by_city = df.groupby('city').agg({'field1':{'mean':'mean', 'std':'std'}, 
																	'review count':'sum', 
																	'business_id':'count'})
#Sorting the df 
#Don't forget that most operations return a copy of your df, not the df itself (unless properly specified)
Sort df by values: df.sort_values('stars') #sorts by stars
Sort df by index: df.set_index('business_id').sort_index()

#Replace, join, merge 

	Replace states by code of state: #1/ for each state, associate code with zip 2/ replace in the df 
	state_abbr = dict(zip(census['state'].unique(), [list_of_codes]))
	census['state'] = census['state'].replace(state_abbr)

	Remove last word from city name: 
	census['city'] = census['city'].apply(lambda x: ' '.join(x.split(' ')[:-1])) #pcq on vire le dernier mot

	Merge dfs on two columns:
	merged_df = df.merge(census, on = ['state','city'], [how = inner, outer, left, right]) #by default, inner 

	Add a bunch of new rows
	new_df = pd.concat([df1,df2]) #must have the same nb of fields

#Working with time series 
#There's an example of read html in the notebook 
#Options for resempling include: mean, median, sum, last, first
#Changing frequencies:
Downsample data / Group rows based on datetime index: df.resample('10AS').mean() #Exactly like a groupby on datetime index: group by 10Y + SPECIFY THE METHOD to group by (here: mean all data for 10Y)
Upsample data: df.resample('10').bfill() / df.resample('10').ffill() #Need bfill or ffill since we don't have data for each quarter = we have to tell pandas how to fill the missing values
#All these capabilities are built on pandas timestamp class / pd.Timestapmt('January 8, 2017')
Select data between two dates: pd.Timestamp('Feb. 11 2016 2:30 am') - pd.Timestamp('2015-08-03 5:13pm') #Note that timestamp will try to adapt these timestamp to the actual time format 
Find indexes between two dates and specify frequency: pd.date_range(start = '1/8/2017', end='3/2/2017', freq = '8') #One record every 8 other days
Find data with specific timestamp using offset: 

	from pandas.tseries.offsets import BDay, Day, BMonthEnd
	a = pd.Timestamp('January 9, 2017') - Day(4) #Find data recorded on 4rth day before
	b = pd.Timestamp('January 9, 2017') - BDay(4)# 4 business days before
	c = pd.Timestamp('January 9, 2017') + BMonthEnd(4) #4 days before month's end

Quick histogram to show frequencies: df['review_count'].apply(np.log).hist(bins = 30)
Quick plot with pandas: pd['Annual Growth Rate'].plot() #See pandas doc on visualization 

#Statistics - Study of randomness
#Statistics: study of how random variables behave in aggregate + use of that behavior to make inferences and arguments 
#Much of the math behind statistical calculations is rigorous but its application to real data often involves making imperfect assumptions.
#What we don't do here: math focus = study different types of random variables, probability distrib functions and taking integrals and computing 
#In real life we don't really know probability distributions = we don't know the underlying truth. 
#When we talk of statistical inference, we either want: 1/ to test some hypothesis 2/ to estimate a parameter
#Mostly, used to describe a group of people or events or measurements: compare temperature to avg (test hypothesis), a change of stoc price to the volatility of the stock (stdev)
#Or estimate what the avg salary of a financier (estimation)

 	#Examples of hypothesis testing: 
 	# - Test if an increase in stock price is significant or random 
 	# - Test if there's a significant difference in salaries between employees with/without advanced degrees
 	# - Test whether there is a significant correlation between the amount of money a customer spent at a store and which ads they've been shown 

 	# Examples of parameter estimation: 
 	# Estimating the avg annual return of a stock 
 	# The variance of salaries for a particular job across companies 
 	# Estimatig the correlation coefficient between annual advertising budget and revenue 
 	# You are going to estimate the thing and state how confident you are about this estimate. 

#Estimating the mean 
#First question: is data biased in some way? 
#Remember loi des grands nombres: the more observations you have, 
#Std of the mean = avg difference bt estimate and the mean 
#Special type of stdev (avg difference between two quantities) - On avg, how much do observed values differ from the mean?
#The bigger the sample, the lower the std of the mean 
#Central limit theorem: formalizes the law of large numbers, saying that the estimate mean tends to tend towards the true mean as the sample size grows
#Even better: says what the std error of the mean should be as a function of the sample size  
#If we bring down the size of the sample, the central limit theorem tells us what distribution of sample means we should expect. 
#The larger the sample size, the better the central limit theorem describes what we'll see.  
#When sample size and nb of samples is small, then distribution often skewed to the left 
#As we increase the samples size, te distirb of the sample means comes to resemble a normal distribution. This occurs regardless of the underlying distribution of the random variable. 
#Std error of the mean = sqrt(var(distrib)/sample size) - as the nb of samples increase, the std error decreases.
#Note: we do talk about two distributions = original distrib (of the underlying random variable) v sample mean (normally distributed when high sample size)
#The sample mean is then its own random variable 

#Hypothesis testing and z-scores
# Null hypothesis is the one we test, and assume to be wrong.
# If we see a difference between sample mean & assumed real mean > std error, means that we got the hypothesis wrong. 
# To test our hypothesis, we need two things: null hyp + threshold to determine what's signifcative evidence that it is wrong = std error might be one stdev away from the sample mean, two stdev... 
# In practice, we choose a propbability alpha (typically small) that the difference between sample mean and true mean is due to random chance. We want it to be small, because 
# we know that the sample mean will be different from the null hypothesis 
# Anyway: if that prob is small, then its not random chance, so the calculated mean is significant.
# we encode the null hypothesis with mu = assumed_true_mean 
# Then collect data and compute sample mean. 
# Then compute the Z-score = difference between sample mean and assumed true mean to the std error. 
# Formula: z= (sample_mean - mu)/stdev/sqrt(n) where stdev = from sample, n = length sample, mu = assumed true mean.
# We get an absolute value (distance/error) that we need to turn into a probability then compare to alpha
# Using the central limit theorem: p_val = sp.stats.norm.cdf(z) #P-value = prob of observing this z_score purely by random chance (we get 0.04 = 4% difference that we'd observe this average salary or smth even more different purely by random chance)
# Then we have significant evidence - and we reject the null hypothesis (our assumed_mean was too high compared to sample mean)

#Confidence Intervals (Central Limit Theorem for parameter estimation)
# If I center my CL distribution on my sample mean, I can calculate how far to either side I need to go to capture 95% of the probability (that the true mean is inside?) Then, there's only 2.5% probability that the true mean is right outside of that range
# Testing hypothesis with confidence intervals: details to check: one-sided or two-sided tests   
# How did we compute the confidence interval: 
# So you need to replace the x by (1+alpha)/2 to get two-sided test bound. Those bounds are essentially z-scores
sp.stats.norm.ppf(x) returns the critical point where captured area < x (x 0.01 for 1% of the area...)







#what does scipy: calculus, numerical derivatives, integrals, lognormal distrib (sp.stats.lognorm(1,loc=20, scale = 25))
#Ar.shape() gives you the shape, ar.reshape(nb rows, nb cols) reshapes the ar