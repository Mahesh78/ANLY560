import pymysql

class sakila(object):
    def __init__(self,add,addVal,ret):
        self.add = add
        self.addVal = addVal
        self.ret = ret
    def conn(self):
        db = pymysql.connect("localhost","user","pass","sakila")
#        cursor = db.cursor()
#        return cursor
        return db
    
    def addRecord(self):
        mydb = self.conn()
        cursor = mydb.cursor()
        ff = cursor
        ff.execute(self.add,self.addVal)
        mydb.commit()
    
    def retrieveRecord(self):
        mydbs = self.conn()
        cursors = mydbs.cursor()
        ff = cursors
        ff.execute(self.ret)
        mydbs.commit()
        result = ff.fetchall()
        print(result)
 
aa = "INSERT INTO actor (actor_id, first_name, last_name, last_update) VALUES (%s,%s, %s, %s)"
ab = ("201","Jonnys","Doe","2010-05-06 00:00:00")
bb = '''(SELECT * FROM actor WHERE first_name = 'Jonnys')'''

a = sakila(aa,ab,bb)
print(' Connection established to the database\n')

print(' Adding a record to the database... \n')
a.addRecord()
print(' Done adding a record to the database!')
print('\n Retrieving newly added record from the database... \n')
a.retrieveRecord()
