def start():
    print("System Started")
    print("States: 1. Solid  |  2. Liquid")
    state = int(input("Enter the state: "))
    if state == 1:
        solid()
    elif state == 2:
        liquid()
    else:
        print("Invalid state choice")


def solid():
    print("Solid types: 1. Rock  |  2. Dust")
    s_type = int(input("Choose solid type: "))
    if s_type == 1:
        print("Shape changed to rectangle")
    elif s_type == 2:
        print("It will break into small particles")
    else:
        print("Invalid solid type")


def liquid():
    print("It is now a liquid state")


def stop():
    print("System Stopped")


def left():
    print("Direction changed → LEFT")


def right():
    print("Direction changed → RIGHT")


def turnoff():
    print("THANK YOU!! Exiting program...")


def main():
    while True:
        print('''\n=== MENU ===
1. Start
2. Stop
3. Left
4. Right
5. Turn off''')

        try:
            choice = int(input("Enter your choice: "))
        except ValueError:
            print("Please enter a number only!")
            continue

        if choice == 1:
            start()
        elif choice == 2:
            stop()
        elif choice == 3:
            left()
        elif choice == 4:
            right()
        elif choice == 5:
            turnoff()
            break
        else:
            print("Invalid menu option")


# Run program
main()

