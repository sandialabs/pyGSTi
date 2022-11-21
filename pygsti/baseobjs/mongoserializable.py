"""
Defines the MongoSerializable class
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
from pygsti.baseobjs.nicelyserializable import NicelySerializable as _NicelySerializable


class MongoSerializable(_NicelySerializable):
    """
    An extension of NicelySerializable for that need to perform special MongoDB serialization.

    An interface that allows an object to save large chunks of data using, e.g.,
    MongoDB's GridFS system, when serializing it as a json-able object for storage
    in a MongoDB database.
    """
    @classmethod
    def from_mongodb_serialization(cls, state, mongodb):
        """
        Create and initialize an object from a "nice" serialization and a auxiliary MongoDB instance.

        A "nice" serialization here means one that is compatible with common text file formats, namely
        JSON (e.g. that dictionary keys must be strings).  This serialization should have been created
        by a prior call to `to_mongodb_serialization` using a reference to the same MongoDB database
        as was used to serialize the object.

        The `state` argument is typically a dictionary containing 'module' and 'state' keys specifying
        the type of object that should be created.  This type must be this class or a subclass of it.

        Parameters
        ----------
        state : object
            An object, usually a dictionary, representing the object to de-serialize.

        mongodb_database : pymongo.database.Database
            An MongoDB instance that may optionally hold auxiliary data associated with the
            object but too large or complex to include in the "nice" serialization itself.

        Returns
        -------
        object
        """
        # Implementation note:
        # This method is similar to _from_nice_serialization, but will defer to the method of a derived class
        # when once is specified in the state dictionary.  This method should thus be used when de-serializing
        # using a potential base class, i.e.  BaseClass._from_nice_serialization_base(state).
        # (This method should rarely need to be overridden in derived (sub) classes.)
        if isinstance(state, dict) and state['module'] == cls.__module__ and state['class'] == cls.__name__:
            # then the state is actually for this class and we should call its _from_nice_serialization method:
            return cls._from_mongodb_serialization(state, mongodb)
        else:
            # otherwise, this call functions as a base class call that defers to the correct derived class
            return MongoSerializable._from_mongodb_serialization.__func__(cls, state, mongodb)

    def to_mongodb_serialization(self, mongodb):
        """
        Serialize this object in a "nice" way, with the optional help of an auxiliary MongoDB instance.

        "Nice" here means that the resulting serialized form is compatible with common text formats,
        namely JSON.  Object components that are especially large or awkward can be stored
        within the provided MongoDB database, and links/ids of the stored documents included in the
        returned serialization.

        Parameters
        ----------
        mongodb_database : pymongo.database.Database
            An auxiliary MongoDB instance to optionally place data within (see above).

        Returns
        -------
        object
            Usually a dictionary representing the serialized object, but may also be another native
            Python type, e.g. a string or list.
        """
        return self._to_mongodb_serialization(mongodb)

    @classmethod
    def _from_mongodb_serialization(cls, state, mongodb):
        c = cls._state_class(state)
        if not issubclass(c, cls):
            raise ValueError("Class '%s' is trying to load a serialized '%s' object (not a subclass)!"
                             % (cls.__module__ + '.' + cls.__name__, state['module'] + '.' + state['class']))
        implementing_cls = cls
        for candidate_cls in c.__mro__:
            if '_from_mongodb_serialization' in candidate_cls.__dict__:
                implementing_cls = candidate_cls; break

        if implementing_cls == cls:  # then there's no actual derived-class implementation to call!
            raise NotImplementedError("Class '%s' doesn't implement _from_mongodb_serialization!"
                                      % str(state['module'] + '.' + state['class']))
        else:
            return c._from_mongodb_serialization(state, mongodb)

    def _to_mongodb_serialization(self, mongodb):
        #Call base class (*not* self.to_nice_serialization) here since this
        # should just behave like the base to_nice_serialization, putting class & module name into a state
        return _NicelySerializable._to_nice_serialization(self)
