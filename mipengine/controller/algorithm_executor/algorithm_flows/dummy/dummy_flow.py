#import run_algorithm #yeah... fix those names...
from AlgorithmExecutor import AlgorithmFlow_abstract
from pprint import pprint
import pdb

from data_classes import TableInfo, TableData,TableView, ColumnInfo, UDFInfo, Parameter

class AlgorithmFlow(AlgorithmFlow_abstract):
    
    def run(self):
        print(f"\n(dummy_flow.py::AlgorithmFlow::run) just in")
        runtime_interface=self.runtime_interface

        return "ok"
