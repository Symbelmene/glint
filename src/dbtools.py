import psycopg2 as pg


def main():
    conn = pg.connect("dbname='findata' user='user' host='0.0.0.0' password='pass' port='5432'")


if __name__ == '__main__':
    main()