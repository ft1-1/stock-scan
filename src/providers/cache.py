"""
Advanced caching layer for provider responses with TTL, size limits, and persistence.

This module provides a comprehensive caching system for API responses with:
- TTL (Time To Live) based expiration
- Memory and disk-based storage options
- Size-based cache eviction (LRU)
- Provider-specific cache strategies
- Cache hit/miss metrics
- Data compression for large responses
- Thread-safe operations
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
import zlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Tuple
from dataclasses import dataclass, field
import threading
from collections import OrderedDict

from src.models import ProviderType, CacheInfo

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Individual cache entry with metadata."""
    data: Any
    created_at: float
    expires_at: float
    hits: int = 0
    size_bytes: int = 0
    compressed: bool = False
    provider: Optional[ProviderType] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_size_bytes: int = 0
    entries_count: int = 0
    evictions_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate


class CacheManager:
    """
    Advanced cache manager with multiple storage backends and intelligent eviction.
    
    Features:
    - Multiple TTL strategies per data type
    - LRU eviction with size limits
    - Optional compression for large data
    - Disk persistence with async I/O
    - Provider-specific cache strategies
    - Performance metrics and monitoring
    """
    
    def __init__(
        self,
        max_memory_size_mb: int = 500,
        cache_dir: Optional[Path] = None,
        enable_compression: bool = True,
        compression_threshold_bytes: int = 1024,
        enable_persistence: bool = True,
        default_ttl_seconds: int = 3600
    ):
        self.max_memory_size_bytes = max_memory_size_mb * 1024 * 1024
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/cache")
        self.enable_compression = enable_compression
        self.compression_threshold_bytes = compression_threshold_bytes
        self.enable_persistence = enable_persistence
        self.default_ttl_seconds = default_ttl_seconds
        
        # Thread-safe cache storage
        self._memory_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = CacheStats()
        
        # TTL strategies by data type
        self.ttl_strategies = {
            'stock_quotes': 300,      # 5 minutes
            'options_chains': 900,    # 15 minutes
            'fundamentals': 86400,    # 24 hours
            'technicals': 3600,       # 1 hour
            'earnings': 86400,        # 24 hours
            'news': 1800,            # 30 minutes
            'screening': 3600,        # 1 hour
            'health_check': 300,      # 5 minutes
        }
        
        # Provider-specific strategies
        self.provider_strategies = {
            ProviderType.EODHD: {
                'max_size_mb': 200,
                'default_ttl': 3600,
                'compress_large_responses': True
            },
            ProviderType.MARKETDATA: {
                'max_size_mb': 200,
                'default_ttl': 900,  # Shorter TTL for more dynamic options data
                'compress_large_responses': True
            },
            ProviderType.CLAUDE: {
                'max_size_mb': 100,
                'default_ttl': 7200,  # Longer TTL for AI analysis
                'compress_large_responses': False  # AI responses typically small
            }
        }
        
        # Ensure cache directory exists
        if self.enable_persistence:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Cache manager initialized: memory_limit={max_memory_size_mb}MB, persistence={enable_persistence}")
    
    async def get(
        self, 
        key: str, 
        provider: Optional[ProviderType] = None,
        data_type: Optional[str] = None
    ) -> Optional[Any]:
        """
        Retrieve data from cache if valid and not expired.
        
        Args:
            key: Cache key
            provider: Provider type for metrics
            data_type: Data type for TTL strategy
            
        Returns:
            Cached data or None if not found/expired
        """
        with self._lock:
            self.stats.total_requests += 1
            
            # Check memory cache first
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                
                # Check if expired
                if time.time() > entry.expires_at:
                    logger.debug(f"Cache entry expired: {key}")
                    del self._memory_cache[key]
                    self.stats.cache_misses += 1
                    return None
                
                # Move to end (LRU)
                self._memory_cache.move_to_end(key)
                entry.hits += 1
                self.stats.cache_hits += 1
                
                # Decompress if needed
                data = self._decompress_data(entry.data) if entry.compressed else entry.data
                
                logger.debug(f"Cache hit: {key} (hits: {entry.hits})")
                return data
            
            self.stats.cache_misses += 1
            
            # Try disk cache if enabled
            if self.enable_persistence:
                disk_data = await self._load_from_disk(key)
                if disk_data is not None:
                    logger.debug(f"Loaded from disk cache: {key}")
                    return disk_data
            
            return None
    
    async def set(
        self,
        key: str,
        data: Any,
        ttl_seconds: Optional[int] = None,
        provider: Optional[ProviderType] = None,
        data_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Store data in cache with specified TTL.
        
        Args:
            key: Cache key
            data: Data to cache
            ttl_seconds: Time to live in seconds
            provider: Provider type for strategy selection
            data_type: Data type for TTL strategy
            tags: Tags for cache entry
        """
        with self._lock:
            # Determine TTL
            if ttl_seconds is None:
                if data_type and data_type in self.ttl_strategies:
                    ttl_seconds = self.ttl_strategies[data_type]
                elif provider and provider in self.provider_strategies:
                    ttl_seconds = self.provider_strategies[provider]['default_ttl']
                else:
                    ttl_seconds = self.default_ttl_seconds
            
            # Serialize and optionally compress data
            serialized_data = pickle.dumps(data)
            compressed = False
            
            if (self.enable_compression and 
                len(serialized_data) > self.compression_threshold_bytes):
                serialized_data = zlib.compress(serialized_data)
                compressed = True
            
            # Create cache entry
            now = time.time()
            entry = CacheEntry(
                data=serialized_data,
                created_at=now,
                expires_at=now + ttl_seconds,
                size_bytes=len(serialized_data),
                compressed=compressed,
                provider=provider,
                tags=tags or []
            )
            
            # Store in memory cache
            self._memory_cache[key] = entry
            self.stats.entries_count += 1
            self.stats.total_size_bytes += entry.size_bytes
            
            # Enforce size limits
            await self._enforce_size_limits()
            
            # Persist to disk if enabled
            if self.enable_persistence:
                await self._save_to_disk(key, entry)
            
            logger.debug(f"Cached: {key} (TTL: {ttl_seconds}s, size: {entry.size_bytes} bytes, compressed: {compressed})")
    
    async def invalidate(self, key: str) -> bool:
        """
        Remove specific key from cache.
        
        Args:
            key: Cache key to remove
            
        Returns:
            True if key was found and removed
        """
        with self._lock:
            removed = key in self._memory_cache
            if removed:
                entry = self._memory_cache.pop(key)
                self.stats.entries_count -= 1
                self.stats.total_size_bytes -= entry.size_bytes
                
                # Remove from disk
                if self.enable_persistence:
                    disk_path = self.cache_dir / f"{self._hash_key(key)}.cache"
                    if disk_path.exists():
                        disk_path.unlink()
                
                logger.debug(f"Invalidated cache entry: {key}")
            
            return removed
    
    async def invalidate_by_tags(self, tags: List[str]) -> int:
        """
        Remove all cache entries with any of the specified tags.
        
        Args:
            tags: List of tags to match
            
        Returns:
            Number of entries removed
        """
        with self._lock:
            keys_to_remove = []
            
            for key, entry in self._memory_cache.items():
                if any(tag in entry.tags for tag in tags):
                    keys_to_remove.append(key)
            
            removed_count = 0
            for key in keys_to_remove:
                if await self.invalidate(key):
                    removed_count += 1
            
            logger.info(f"Invalidated {removed_count} cache entries with tags: {tags}")
            return removed_count
    
    async def invalidate_by_provider(self, provider: ProviderType) -> int:
        """
        Remove all cache entries for a specific provider.
        
        Args:
            provider: Provider type to invalidate
            
        Returns:
            Number of entries removed
        """
        with self._lock:
            keys_to_remove = []
            
            for key, entry in self._memory_cache.items():
                if entry.provider == provider:
                    keys_to_remove.append(key)
            
            removed_count = 0
            for key in keys_to_remove:
                if await self.invalidate(key):
                    removed_count += 1
            
            logger.info(f"Invalidated {removed_count} cache entries for provider: {provider}")
            return removed_count
    
    async def clear_expired(self) -> int:
        """
        Remove all expired cache entries.
        
        Returns:
            Number of entries removed
        """
        with self._lock:
            now = time.time()
            keys_to_remove = []
            
            for key, entry in self._memory_cache.items():
                if now > entry.expires_at:
                    keys_to_remove.append(key)
            
            removed_count = 0
            for key in keys_to_remove:
                if await self.invalidate(key):
                    removed_count += 1
            
            if removed_count > 0:
                logger.info(f"Cleared {removed_count} expired cache entries")
            
            return removed_count
    
    async def clear_all(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of entries removed
        """
        with self._lock:
            removed_count = len(self._memory_cache)
            self._memory_cache.clear()
            self.stats = CacheStats()  # Reset stats
            
            # Clear disk cache
            if self.enable_persistence and self.cache_dir.exists():
                for cache_file in self.cache_dir.glob("*.cache"):
                    cache_file.unlink()
            
            logger.info(f"Cleared all {removed_count} cache entries")
            return removed_count
    
    def get_stats(self) -> CacheStats:
        """Get current cache statistics."""
        with self._lock:
            # Update current stats
            self.stats.total_size_bytes = sum(entry.size_bytes for entry in self._memory_cache.values())
            self.stats.entries_count = len(self._memory_cache)
            return self.stats
    
    def get_cache_info(self, key: str) -> Optional[CacheInfo]:
        """Get metadata about a cache entry."""
        with self._lock:
            if key not in self._memory_cache:
                return None
            
            entry = self._memory_cache[key]
            return CacheInfo(
                cache_key=key,
                provider=entry.provider or ProviderType.EODHD,  # Default fallback
                cached_at=datetime.fromtimestamp(entry.created_at),
                expires_at=datetime.fromtimestamp(entry.expires_at),
                ttl_seconds=int(entry.expires_at - entry.created_at),
                data_size_bytes=entry.size_bytes,
                hit_count=entry.hits,
                is_stale=time.time() > entry.expires_at
            )
    
    async def _enforce_size_limits(self) -> None:
        """Enforce memory size limits using LRU eviction."""
        while self.stats.total_size_bytes > self.max_memory_size_bytes and self._memory_cache:
            # Remove least recently used entry
            key, entry = self._memory_cache.popitem(last=False)
            self.stats.total_size_bytes -= entry.size_bytes
            self.stats.entries_count -= 1
            self.stats.evictions_count += 1
            
            # Remove from disk if exists
            if self.enable_persistence:
                disk_path = self.cache_dir / f"{self._hash_key(key)}.cache"
                if disk_path.exists():
                    disk_path.unlink()
            
            logger.debug(f"Evicted cache entry (LRU): {key}")
    
    async def _save_to_disk(self, key: str, entry: CacheEntry) -> None:
        """Save cache entry to disk asynchronously."""
        try:
            disk_path = self.cache_dir / f"{self._hash_key(key)}.cache"
            
            # Create metadata
            metadata = {
                'key': key,
                'created_at': entry.created_at,
                'expires_at': entry.expires_at,
                'hits': entry.hits,
                'compressed': entry.compressed,
                'provider': entry.provider.value if entry.provider else None,
                'tags': entry.tags
            }
            
            # Save both metadata and data
            with open(disk_path, 'wb') as f:
                pickle.dump({
                    'metadata': metadata,
                    'data': entry.data
                }, f)
            
        except Exception as e:
            logger.warning(f"Failed to save cache entry to disk: {key} - {e}")
    
    async def _load_from_disk(self, key: str) -> Optional[Any]:
        """Load cache entry from disk if valid."""
        try:
            disk_path = self.cache_dir / f"{self._hash_key(key)}.cache"
            if not disk_path.exists():
                return None
            
            with open(disk_path, 'rb') as f:
                cached = pickle.load(f)
            
            metadata = cached['metadata']
            data = cached['data']
            
            # Check if expired
            if time.time() > metadata['expires_at']:
                disk_path.unlink()  # Remove expired file
                return None
            
            # Add back to memory cache
            entry = CacheEntry(
                data=data,
                created_at=metadata['created_at'],
                expires_at=metadata['expires_at'],
                hits=metadata['hits'],
                compressed=metadata['compressed'],
                provider=ProviderType(metadata['provider']) if metadata['provider'] else None,
                tags=metadata['tags']
            )
            
            self._memory_cache[key] = entry
            
            # Decompress if needed
            return self._decompress_data(data) if entry.compressed else pickle.loads(data)
            
        except Exception as e:
            logger.warning(f"Failed to load cache entry from disk: {key} - {e}")
            return None
    
    def _hash_key(self, key: str) -> str:
        """Generate consistent hash for cache key."""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _decompress_data(self, data: bytes) -> Any:
        """Decompress and deserialize data."""
        try:
            decompressed = zlib.decompress(data)
            return pickle.loads(decompressed)
        except Exception as e:
            logger.error(f"Failed to decompress cache data: {e}")
            raise


class ProviderCacheWrapper:
    """
    Wrapper that provides caching for provider methods with automatic key generation.
    
    This wrapper automatically handles:
    - Cache key generation from method parameters
    - Provider-specific caching strategies
    - Automatic cache invalidation
    - Method call interception
    """
    
    def __init__(self, provider, cache_manager: CacheManager):
        self.provider = provider
        self.cache_manager = cache_manager
        self.provider_type = getattr(provider, 'provider_type', ProviderType.EODHD)
    
    async def get_stock_quote(self, symbol: str) -> Any:
        """Cached wrapper for get_stock_quote."""
        cache_key = f"stock_quote:{self.provider_type.value}:{symbol}"
        
        # Try cache first
        cached_data = await self.cache_manager.get(
            cache_key, 
            provider=self.provider_type,
            data_type='stock_quotes'
        )
        
        if cached_data is not None:
            return cached_data
        
        # Fetch from provider
        data = await self.provider.get_stock_quote(symbol)
        
        # Cache the result
        if data is not None:
            await self.cache_manager.set(
                cache_key,
                data,
                provider=self.provider_type,
                data_type='stock_quotes',
                tags=[f'symbol:{symbol}', 'stock_quotes']
            )
        
        return data
    
    async def get_options_chain(self, symbol: str, **filters) -> Any:
        """Cached wrapper for get_options_chain."""
        # Create cache key from symbol and filters
        filter_hash = hashlib.md5(json.dumps(filters, sort_keys=True).encode()).hexdigest()[:8]
        cache_key = f"options_chain:{self.provider_type.value}:{symbol}:{filter_hash}"
        
        # Try cache first
        cached_data = await self.cache_manager.get(
            cache_key,
            provider=self.provider_type,
            data_type='options_chains'
        )
        
        if cached_data is not None:
            return cached_data
        
        # Fetch from provider
        data = await self.provider.get_options_chain(symbol, **filters)
        
        # Cache the result
        if data is not None:
            await self.cache_manager.set(
                cache_key,
                data,
                provider=self.provider_type,
                data_type='options_chains',
                tags=[f'symbol:{symbol}', 'options_chains']
            )
        
        return data
    
    async def get_fundamental_data(self, symbol: str) -> Any:
        """Cached wrapper for get_fundamental_data."""
        cache_key = f"fundamentals:{self.provider_type.value}:{symbol}"
        
        # Try cache first
        cached_data = await self.cache_manager.get(
            cache_key,
            provider=self.provider_type,
            data_type='fundamentals'
        )
        
        if cached_data is not None:
            return cached_data
        
        # Fetch from provider
        data = await self.provider.get_fundamental_data(symbol)
        
        # Cache the result
        if data is not None:
            await self.cache_manager.set(
                cache_key,
                data,
                provider=self.provider_type,
                data_type='fundamentals',
                tags=[f'symbol:{symbol}', 'fundamentals']
            )
        
        return data


# Global cache manager instance
_global_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get or create global cache manager instance."""
    global _global_cache_manager
    
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager()
    
    return _global_cache_manager


def initialize_cache_manager(config: Dict[str, Any]) -> CacheManager:
    """Initialize global cache manager with configuration."""
    global _global_cache_manager
    
    _global_cache_manager = CacheManager(
        max_memory_size_mb=config.get('max_cache_size_mb', 500),
        cache_dir=Path(config.get('cache_directory', 'data/cache')),
        enable_compression=config.get('enable_compression', True),
        enable_persistence=config.get('enable_persistence', True),
        default_ttl_seconds=config.get('cache_ttl', 3600)
    )
    
    return _global_cache_manager