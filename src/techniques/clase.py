
"""
Clase_prueba
============

"""

class Client:
    """Representa un cliente.

    :param client_num: Número de cliente.
    :param dni: DNI del cliente.
    :param name: Nombre del cliente.
    :param phone: Teléfono del cliente.
    :param boats: Barcos del cliente.

    :type client_num: string
    :type dni: string
    :type name: string
    :type phone: string
    :type boats: dict
    """

    def __init__(self, client_num=None, dni=None, name=None, phone=None, boats={}):
        """Inicializa un objeto de la clase Cliente."""
        self.client_num = client_num
        self.dni = dni
        self.name = name
        self.phone = phone
        self.boats = boats

    @property
    def dni(self):
        """
        Dni del cliente.

        :getter: Devuelve el dni del cliente
        :setter: Comprueba el dni y si no es correcto,lanza una excepción

        :type: string
        """
        return self.__dni

    def add_new_boat(self, boat):
        """
        Añade un nuevo barco a la lista de barcos del cliente.

        :param boat: Un objeto barco.

        :type boat: Boat
        """
        self.boats[boat.matricula] = boat

    def remove_boat(self, boat):
        """
        Elimina un barco de la lista de barcos del cliente.

        :param boat: Un objeto barco.

        :type boat: Boat
        """
        del self.barcos[boat.matricula]

    def __str__(self):
        """Reescribe el método __str__"""
        return self.client_num + ", " + self.name

    def show_details(self):
        """Muestra por pantalla los datos de un cliente"""
        print(f"Nº cliente: {self.client_num}")
        print(f"DNI: {self.dni}")
        print(f"Nombre: {self.name}")
        if self.phone is not None:
            print(f"Teléfono: {self.phone}")
