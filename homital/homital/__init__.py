import pymysql

# Make PyMySQL act as the MySQLdb driver that Django's mysql backend expects
pymysql.install_as_MySQLdb()
