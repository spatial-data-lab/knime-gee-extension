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
