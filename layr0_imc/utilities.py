# -*- coding: utf-8 -*-
"""
layr0_imc REST API Documentation - Utilities Methods
    https://docs.layr0.org
"""

from datetime import datetime
from .base import BaseAPI


class UtilitiesAPI(BaseAPI):
    """
    Utilities API methods for layr0_imc.
    Inherits from the BaseAPI class.
    """

    def holidays(self, year=None):
        """
        Get market holidays for a specific year.

        Args:
            year (int, optional): Year for holiday data (2020-2050).
                                  Defaults to current year if not provided.

        Returns:
            dict: Response containing holiday data with the following structure:
                {
                    "status": "success",
                    "year": 2025,
                    "timezone": "Asia/Kolkata",
                    "data": [
                        {
                            "date": "2025-02-26",
                            "description": "Maha Shivaratri",
                            "holiday_type": "TRADING_HOLIDAY",
                            "closed_exchanges": ["NSE", "BSE", "NFO", "BFO", "CDS", "BCD"],
                            "open_exchanges": [
                                {
                                    "exchange": "MCX",
                                    "start_time": 1740549000000,
                                    "end_time": 1740602700000
                                }
                            ]
                        }
                    ]
                }
        """
        payload = {
            "apikey": self.api_key
        }

        if year is not None:
            payload["year"] = year

        return self._make_request("market/holidays", payload)

    def timings(self, date=None):
        """
        Get market timings for a specific date.

        Args:
            date (str, optional): Date in YYYY-MM-DD format.
                                  Defaults to current date if not provided.

        Returns:
            dict: Response containing market timing data with the following structure:
                {
                    "status": "success",
                    "data": [
                        {
                            "exchange": "NSE",
                            "start_time": 1745984700000,
                            "end_time": 1746007200000
                        },
                        {
                            "exchange": "BSE",
                            "start_time": 1745984700000,
                            "end_time": 1746007200000
                        },
                        {
                            "exchange": "MCX",
                            "start_time": 1745983800000,
                            "end_time": 1746037500000
                        }
                    ]
                }

                Note: Empty data array indicates market is closed (holiday).
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        payload = {
            "apikey": self.api_key,
            "date": date
        }

        return self._make_request("market/timings", payload)
