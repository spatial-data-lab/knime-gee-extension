# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------
#  Copyright by KNIME AG, Zurich, Switzerland
#  Website: http://www.knime.com; Email: contact@knime.com
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License, Version 3, as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, see <http://www.gnu.org/licenses>.
#
#  Additional permission under GNU GPL version 3 section 7:
#
#  KNIME interoperates with ECLIPSE solely via ECLIPSE's plug-in APIs.
#  Hence, KNIME and ECLIPSE are both independent programs and are not
#  derived from each other. Should, however, the interpretation of the
#  GNU GPL Version 3 ("License") under any applicable laws result in
#  KNIME and ECLIPSE being a combined program, KNIME AG herewith grants
#  you the additional permission to use and propagate KNIME together with
#  ECLIPSE with only the license terms in place for ECLIPSE applying to
#  ECLIPSE and the GNU GPL Version 3 applying for KNIME, provided the
#  license terms of ECLIPSE themselves allow for the respective use and
#  propagation of ECLIPSE together with KNIME.
#
#  Additional permission relating to nodes for KNIME that extend the Node
#  Extension (and in particular that are based on subclasses of NodeModel,
#  NodeDialog, and NodeView) and that only interoperate with KNIME through
#  standard APIs ("Nodes"):
#  Nodes are deemed to be separate and independent programs and to not be
#  covered works.  Notwithstanding anything to the contrary in the
#  License, the License does not apply to Nodes, you are not required to
#  license Nodes under the License, and you are granted a license to
#  prepare and propagate Nodes, in each case even if such Nodes are
#  propagated with or for interoperation with KNIME.  The owner of a Node
#  may freely choose the license terms applicable to such Node, including
#  when such Node is propagated with or for interoperation with KNIME.
# ------------------------------------------------------------------------


import knime.extension as knext
import logging
from knime.extension.nodes import ConnectionPortObject

LOGGER = logging.getLogger(__name__)


class GoogleEarthEngineObjectSpec(knext.PortObjectSpec):
    def __init__(self, project_id: str) -> None:
        super().__init__()
        self._project_id = project_id

    @property
    def project_id(self) -> str:
        return self._project_id

    def serialize(self) -> dict:
        return {"project_id": self._project_id}

    @classmethod
    def deserialize(cls, data: dict) -> "GoogleEarthEngineObjectSpec":
        return cls(data["project_id"])


# Since this is a connection port object, all nodes connected to this port use the same Python process.
# Therefore, initializing the Google Earth Engine in the Google Earth Engine Connector node is enough.
# Any stateful objects created there will be shared with all other nodes connected to it.
class GoogleEarthEngineConnectionObject(ConnectionPortObject):
    def __init__(
        self,
        spec: GoogleEarthEngineObjectSpec,
        credentials=None,
        gee_object=None,  # add：GEE Object
        classifier=None,  # add：Classifier Object
        reverse_mapping=None,  # add：Reverse mapping for class values
    ) -> None:
        super().__init__(spec)
        self._credentials = credentials
        self._gee_object = gee_object  # add：GEE Object
        self._classifier = classifier  # add：Classifier Object
        self._reverse_mapping = reverse_mapping  # add：Reverse mapping

    @property
    def spec(self) -> GoogleEarthEngineObjectSpec:
        return super().spec

    @property
    def credentials(self):
        return self._credentials

    @property
    def gee_object(self):  # add：GEE Object
        return self._gee_object

    @property
    def classifier(self):  # add：Classifier Object
        """Get stored classifier object (ee.Classifier)"""
        return self._classifier

    @property
    def reverse_mapping(self):
        """Get reverse mapping for class values (maps 0-based indices back to original class values)"""
        return self._reverse_mapping

    def to_connection_data(self):
        """
        Provide the data that makes up this ConnectionPortObject such that it can be used
        by downstream nodes in the ``from_connection_data`` method.
        """
        return {
            "credentials": self._credentials,
            "gee_object": self._gee_object,  # add：transfer GEE Object
            "classifier": self._classifier,  # add：transfer Classifier Object
            "reverse_mapping": self._reverse_mapping,  # add：transfer reverse mapping
        }

    @classmethod
    def from_connection_data(
        cls, spec: knext.PortObjectSpec, data
    ) -> "ConnectionPortObject":
        """
        Construct a ConnectionPortObject from spec and data. The data is the data that has
        been returned by the ``to_connection_data`` method of the ConnectionPortObject
        by the upstream node.

        The data should not be tempered with, as it is a Python object that is handed to
        all nodes using this ConnectionPortObject.
        """
        credentials = data.get("credentials") if data else None
        gee_object = data.get("gee_object") if data else None  # add：GEE Object
        classifier = data.get("classifier") if data else None  # add：Classifier Object
        reverse_mapping = (
            data.get("reverse_mapping") if data else None
        )  # add：Reverse mapping
        return cls(
            spec, credentials, gee_object, classifier, reverse_mapping
        )  # add：transfer Objects


google_earth_engine_port_type = knext.port_type(
    "Google Earth Engine Port Type",
    GoogleEarthEngineConnectionObject,
    GoogleEarthEngineObjectSpec,
)


############################################
# Specialized Connection Objects
############################################


# Base class for all GEE connection objects
# Since this is a connection port object, all nodes connected to this port use the same Python process.
# Therefore, initializing the Google Earth Engine in the Google Earth Engine Connector node is enough.
# Any stateful objects created there will be shared with all other nodes connected to it.
class BaseGEEConnectionObject(ConnectionPortObject):
    """Base class for all GEE connection objects with shared credentials and project_id."""

    def __init__(self, spec: GoogleEarthEngineObjectSpec, credentials=None):
        super().__init__(spec)
        self._credentials = credentials

    @property
    def spec(self) -> GoogleEarthEngineObjectSpec:
        return super().spec

    @property
    def credentials(self):
        return self._credentials


# Image Connection Object
class GEEImageConnectionObject(BaseGEEConnectionObject):
    """Connection object specifically for GEE Image objects."""

    def __init__(self, spec: GoogleEarthEngineObjectSpec, credentials=None, image=None):
        super().__init__(spec, credentials)
        self._image = image

    @property
    def image(self):
        """Get the GEE Image object"""
        return self._image

    def to_connection_data(self):
        return {
            "credentials": self._credentials,
            "image": self._image,
        }

    @classmethod
    def from_connection_data(cls, spec: knext.PortObjectSpec, data):
        credentials = data.get("credentials") if data else None
        image = data.get("image") if data else None
        return cls(spec, credentials, image)


# Feature Collection Connection Object
class GEEFeatureCollectionConnectionObject(BaseGEEConnectionObject):
    """Connection object specifically for GEE FeatureCollection objects."""

    def __init__(
        self,
        spec: GoogleEarthEngineObjectSpec,
        credentials=None,
        feature_collection=None,
    ):
        super().__init__(spec, credentials)
        self._feature_collection = feature_collection

    @property
    def feature_collection(self):
        """Get the GEE FeatureCollection object"""
        return self._feature_collection

    def to_connection_data(self):
        return {
            "credentials": self._credentials,
            "feature_collection": self._feature_collection,
        }

    @classmethod
    def from_connection_data(cls, spec: knext.PortObjectSpec, data):
        credentials = data.get("credentials") if data else None
        feature_collection = data.get("feature_collection") if data else None
        return cls(spec, credentials, feature_collection)


# Classifier Connection Object
class GEEClassifierConnectionObject(BaseGEEConnectionObject):
    """Connection object specifically for GEE Classifier objects with training metadata."""

    def __init__(
        self,
        spec: GoogleEarthEngineObjectSpec,
        credentials=None,
        classifier=None,
        training_data=None,  # Training data (already remapped)
        label_property=None,  # Label property name
        reverse_mapping=None,  # Reverse mapping for class values
        input_properties=None,  # Input properties (bands) used during training
    ):
        super().__init__(spec, credentials)
        self._classifier = classifier
        self._training_data = training_data
        self._label_property = label_property
        self._reverse_mapping = reverse_mapping
        self._input_properties = input_properties

    @property
    def classifier(self):
        """Get the trained GEE Classifier object"""
        return self._classifier

    @property
    def training_data(self):
        """Get the training data FeatureCollection (already remapped)"""
        return self._training_data

    @property
    def label_property(self):
        """Get the label property name used during training"""
        return self._label_property

    @property
    def reverse_mapping(self):
        """Get reverse mapping for class values (maps 0-based indices back to original class values)"""
        return self._reverse_mapping

    @property
    def input_properties(self):
        """Get the input properties (bands/features) used during training"""
        return self._input_properties

    def to_connection_data(self):
        return {
            "credentials": self._credentials,
            "classifier": self._classifier,
            "training_data": self._training_data,
            "label_property": self._label_property,
            "reverse_mapping": self._reverse_mapping,
            "input_properties": self._input_properties,
        }

    @classmethod
    def from_connection_data(cls, spec: knext.PortObjectSpec, data):
        credentials = data.get("credentials") if data else None
        classifier = data.get("classifier") if data else None
        training_data = data.get("training_data") if data else None
        label_property = data.get("label_property") if data else None
        reverse_mapping = data.get("reverse_mapping") if data else None
        input_properties = data.get("input_properties") if data else None
        return cls(
            spec,
            credentials,
            classifier,
            training_data,
            label_property,
            reverse_mapping,
            input_properties,
        )


# Clusterer Connection Object
class GEEClustererConnectionObject(BaseGEEConnectionObject):
    """Connection object specifically for GEE Clusterer objects with training metadata."""

    def __init__(
        self,
        spec: GoogleEarthEngineObjectSpec,
        credentials=None,
        clusterer=None,
        input_properties=None,  # Input properties (bands) used during training
    ):
        super().__init__(spec, credentials)
        self._clusterer = clusterer
        self._input_properties = input_properties

    @property
    def clusterer(self):
        """Get the trained GEE Clusterer object"""
        return self._clusterer

    @property
    def gee_object(self):
        """Alias for clusterer to maintain compatibility"""
        return self._clusterer

    @property
    def input_properties(self):
        """Get the input properties (bands/features) used during training"""
        return self._input_properties

    def to_connection_data(self):
        return {
            "credentials": self._credentials,
            "clusterer": self._clusterer,
            "input_properties": self._input_properties,
        }

    @classmethod
    def from_connection_data(cls, spec: knext.PortObjectSpec, data):
        credentials = data.get("credentials") if data else None
        clusterer = data.get("clusterer") if data else None
        input_properties = data.get("input_properties") if data else None
        return cls(
            spec,
            credentials,
            clusterer,
            input_properties,
        )


# Specialized Spec classes for different port types
# Each port type needs a unique Spec class, even if they contain the same data
class GEEImageObjectSpec(GoogleEarthEngineObjectSpec):
    """Spec class for GEE Image port type."""

    pass


class GEEFeatureCollectionObjectSpec(GoogleEarthEngineObjectSpec):
    """Spec class for GEE FeatureCollection port type."""

    pass


class GEEClassifierObjectSpec(GoogleEarthEngineObjectSpec):
    """Spec class for GEE Classifier port type."""

    pass


class GEEClustererObjectSpec(GoogleEarthEngineObjectSpec):
    """Spec class for GEE Clusterer port type."""

    pass


class GEEImageCollectionObjectSpec(GoogleEarthEngineObjectSpec):
    """Spec class for GEE ImageCollection port type."""

    pass


# Image Collection Connection Object
class GEEImageCollectionConnectionObject(BaseGEEConnectionObject):
    """Connection object specifically for GEE ImageCollection objects."""

    def __init__(
        self,
        spec: GoogleEarthEngineObjectSpec,
        credentials=None,
        image_collection=None,
    ):
        super().__init__(spec, credentials)
        self._image_collection = image_collection

    @property
    def image_collection(self):
        """Get the GEE ImageCollection object"""
        return self._image_collection

    def to_connection_data(self):
        return {
            "credentials": self._credentials,
            "image_collection": self._image_collection,
        }

    @classmethod
    def from_connection_data(cls, spec: knext.PortObjectSpec, data):
        credentials = data.get("credentials") if data else None
        image_collection = data.get("image_collection") if data else None
        return cls(spec, credentials, image_collection)


# Port types for specialized connection objects
gee_image_port_type = knext.port_type(
    "GEE Image Port Type",
    GEEImageConnectionObject,
    GEEImageObjectSpec,
)

gee_feature_collection_port_type = knext.port_type(
    "GEE Feature Collection Port Type",
    GEEFeatureCollectionConnectionObject,
    GEEFeatureCollectionObjectSpec,
)

gee_classifier_port_type = knext.port_type(
    "GEE Classifier Port Type",
    GEEClassifierConnectionObject,
    GEEClassifierObjectSpec,
)

gee_clusterer_port_type = knext.port_type(
    "GEE Clusterer Port Type",
    GEEClustererConnectionObject,
    GEEClustererObjectSpec,
)

gee_image_collection_port_type = knext.port_type(
    "GEE Image Collection Port Type",
    GEEImageCollectionConnectionObject,
    GEEImageCollectionObjectSpec,
)
