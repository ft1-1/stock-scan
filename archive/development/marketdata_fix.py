# Fixed MarketData options chain method
# Copy this to src/providers/marketdata_client.py


    async def get_options_chain_fixed(
        self,
        symbol: str,
        expiration_date: Optional[str] = None,
        **filters
    ) -> List[OptionContract]:
        """
        Get options chain with proper configuration for cached data.
        FIXED VERSION: Doesn't send feed parameter to get cached data.
        """
        try:
            if not symbol:
                raise ValueError("Symbol is required")
            
            url = f"{self.base_url}/options/chain/{symbol}/"
            
            # Build query parameters
            params = {}
            
            # DO NOT SET feed parameter - let API use cached by default
            # This gives us status 203 with cached data
            
            if expiration_date:
                params['expiration'] = expiration_date
            
            # Apply other filters
            if 'side' in filters:
                params['side'] = filters['side']
            if 'strike' in filters:
                params['strike'] = filters['strike']
            if 'minOpenInterest' in filters:
                params['minOpenInterest'] = filters['minOpenInterest']
            if 'moneyness' in filters:
                params['moneyness'] = filters['moneyness']
            
            # Make request
            response = await self._make_request(
                "GET",
                url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                params=params
            )
            
            if not response.success:
                logger.error(f"Failed to get options chain: {response.error}")
                return []
            
            # Parse response - MarketData returns parallel arrays
            data = response.data
            if data.get('s') != 'ok':
                return []
            
            # Convert arrays to OptionContract objects
            contracts = []
            strikes = data.get('strike', [])
            
            for i in range(len(strikes)):
                contract = OptionContract(
                    option_symbol=data.get('optionSymbol', [])[i] if i < len(data.get('optionSymbol', [])) else f"{symbol}_{strikes[i]}",
                    underlying_symbol=symbol,
                    strike=strikes[i],
                    expiration_date=datetime.fromtimestamp(data.get('expiration', [])[i]) if i < len(data.get('expiration', [])) else None,
                    option_type=OptionType.CALL if data.get('side', [])[i] == 'call' else OptionType.PUT,
                    bid=data.get('bid', [])[i] if i < len(data.get('bid', [])) else 0,
                    ask=data.get('ask', [])[i] if i < len(data.get('ask', [])) else 0,
                    last=data.get('last', [])[i] if i < len(data.get('last', [])) else 0,
                    volume=data.get('volume', [])[i] if i < len(data.get('volume', [])) else 0,
                    open_interest=data.get('openInterest', [])[i] if i < len(data.get('openInterest', [])) else 0,
                    implied_volatility=data.get('iv', [])[i] if i < len(data.get('iv', [])) else 0,
                    delta=data.get('delta', [])[i] if i < len(data.get('delta', [])) else None,
                    gamma=data.get('gamma', [])[i] if i < len(data.get('gamma', [])) else None,
                    theta=data.get('theta', [])[i] if i < len(data.get('theta', [])) else None,
                    vega=data.get('vega', [])[i] if i < len(data.get('vega', [])) else None,
                )
                contracts.append(contract)
            
            # Log success
            logger.info(f"Retrieved {len(contracts)} option contracts for {symbol}")
            
            # Track credit usage (1 credit for cached request)
            self._daily_credits_used += 1
            
            return contracts
            
        except Exception as e:
            logger.error(f"Error getting options chain: {e}")
            return []
