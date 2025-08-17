#!/usr/bin/env python3
"""
Storage Optimizer - Automated Storage Cleanup and Optimization

This script implements comprehensive storage optimization strategies including:
- Log file rotation and compression
- Temporary file cleanup  
- Database optimization
- Cache management
- Storage lifecycle policies
"""

import os
import gzip
import shutil
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import json


class StorageOptimizer:
    """
    Automated storage optimization and cleanup
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        self.logger = self._setup_logging()
        self.optimization_results = []
        
    def _get_default_config(self) -> Dict:
        """Get default optimization configuration"""
        return {
            "log_retention_days": 7,
            "log_compression_days": 1,
            "temp_cleanup_hours": 24,
            "cache_max_size_mb": 100,
            "database_vacuum_days": 7,
            "archive_after_days": 30,
            "delete_after_days": 365
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for optimization operations"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
        
    def optimize_storage(self) -> Dict:
        """Run comprehensive storage optimization"""
        self.logger.info("🚀 Starting storage optimization...")
        
        start_time = datetime.now()
        initial_usage = self._get_storage_usage("/app")
        
        try:
            # 1. Log file optimization
            log_savings = self._optimize_log_files()
            
            # 2. Temporary file cleanup
            temp_savings = self._cleanup_temporary_files()
            
            # 3. Database optimization
            db_savings = self._optimize_databases()
            
            # 4. Cache cleanup
            cache_savings = self._optimize_caches()
            
            # 5. Archive old files
            archive_savings = self._archive_old_files()
            
            # Calculate results
            final_usage = self._get_storage_usage("/app")
            total_savings = initial_usage - final_usage
            duration = (datetime.now() - start_time).total_seconds()
            
            results = {
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": duration,
                "initial_usage_mb": initial_usage / (1024 * 1024),
                "final_usage_mb": final_usage / (1024 * 1024),
                "total_savings_mb": total_savings / (1024 * 1024),
                "savings_percentage": (total_savings / initial_usage) * 100 if initial_usage > 0 else 0,
                "optimizations": {
                    "log_files": log_savings,
                    "temporary_files": temp_savings,
                    "databases": db_savings,
                    "caches": cache_savings,
                    "archives": archive_savings
                },
                "estimated_monthly_cost_savings": self._calculate_cost_savings(total_savings)
            }
            
            self.optimization_results.append(results)
            self.logger.info(f"✅ Storage optimization completed - saved {total_savings/(1024*1024):.1f}MB")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Storage optimization failed: {e}")
            raise
            
    def _optimize_log_files(self) -> Dict:
        """Optimize log files through rotation and compression"""
        self.logger.info("📋 Optimizing log files...")
        
        savings = 0
        files_processed = 0
        files_compressed = 0
        files_deleted = 0
        
        log_dirs = ["/app/logs", "/app/data"]
        
        for log_dir in log_dirs:
            if not os.path.exists(log_dir):
                continue
                
            for root, dirs, files in os.walk(log_dir):
                for file in files:
                    if self._is_log_file(file):
                        filepath = os.path.join(root, file)
                        file_age_days = self._get_file_age_days(filepath)
                        file_size = os.path.getsize(filepath)
                        
                        files_processed += 1
                        
                        # Compress files older than 1 day
                        if (file_age_days > self.config["log_compression_days"] and 
                            not filepath.endswith('.gz')):
                            
                            compressed_path = filepath + '.gz'
                            try:
                                with open(filepath, 'rb') as f_in:
                                    with gzip.open(compressed_path, 'wb') as f_out:
                                        shutil.copyfileobj(f_in, f_out)
                                
                                # Remove original if compression successful
                                if os.path.exists(compressed_path):
                                    os.remove(filepath)
                                    compressed_size = os.path.getsize(compressed_path)
                                    savings += file_size - compressed_size
                                    files_compressed += 1
                                    
                            except Exception as e:
                                self.logger.warning(f"Failed to compress {filepath}: {e}")
                        
                        # Delete files older than retention period
                        elif file_age_days > self.config["log_retention_days"]:
                            try:
                                os.remove(filepath)
                                savings += file_size
                                files_deleted += 1
                            except Exception as e:
                                self.logger.warning(f"Failed to delete {filepath}: {e}")
        
        return {
            "files_processed": files_processed,
            "files_compressed": files_compressed,
            "files_deleted": files_deleted,
            "bytes_saved": savings,
            "mb_saved": savings / (1024 * 1024)
        }
        
    def _cleanup_temporary_files(self) -> Dict:
        """Clean up temporary and cache files"""
        self.logger.info("🗑️ Cleaning temporary files...")
        
        savings = 0
        files_deleted = 0
        
        temp_dirs = ["/tmp", "/app/tmp", "/app/__pycache__"]
        temp_patterns = ["*.tmp", "*.temp", "*.cache", "*.pyc", "*~", ".DS_Store"]
        
        for temp_dir in temp_dirs:
            if not os.path.exists(temp_dir):
                continue
                
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    filepath = os.path.join(root, file)
                    
                    # Check if file matches temp patterns or is old
                    is_temp = any(self._matches_pattern(file, pattern) for pattern in temp_patterns)
                    is_old = self._get_file_age_hours(filepath) > self.config["temp_cleanup_hours"]
                    
                    if is_temp or is_old:
                        try:
                            file_size = os.path.getsize(filepath)
                            os.remove(filepath)
                            savings += file_size
                            files_deleted += 1
                        except Exception as e:
                            self.logger.warning(f"Failed to delete {filepath}: {e}")
        
        return {
            "files_deleted": files_deleted,
            "bytes_saved": savings,
            "mb_saved": savings / (1024 * 1024)
        }
        
    def _optimize_databases(self) -> Dict:
        """Optimize database files"""
        self.logger.info("🗄️ Optimizing databases...")
        
        savings = 0
        databases_optimized = 0
        
        # Find SQLite databases
        for root, dirs, files in os.walk("/app"):
            for file in files:
                if file.endswith('.db') or file.endswith('.sqlite'):
                    filepath = os.path.join(root, file)
                    
                    try:
                        initial_size = os.path.getsize(filepath)
                        
                        # Vacuum database to reclaim space
                        conn = sqlite3.connect(filepath)
                        conn.execute("VACUUM")
                        conn.close()
                        
                        final_size = os.path.getsize(filepath)
                        db_savings = initial_size - final_size
                        savings += db_savings
                        databases_optimized += 1
                        
                        self.logger.info(f"Optimized {file}: saved {db_savings/(1024*1024):.1f}MB")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to optimize {filepath}: {e}")
        
        return {
            "databases_optimized": databases_optimized,
            "bytes_saved": savings,
            "mb_saved": savings / (1024 * 1024)
        }
        
    def _optimize_caches(self) -> Dict:
        """Optimize cache directories"""
        self.logger.info("💾 Optimizing caches...")
        
        savings = 0
        files_deleted = 0
        
        cache_dirs = [
            "/app/__pycache__",
            "/app/.cache",
            "/app/cache",
            "/app/data/cache"
        ]
        
        for cache_dir in cache_dirs:
            if not os.path.exists(cache_dir):
                continue
                
            # Calculate cache size
            cache_size = self._get_directory_size(cache_dir)
            max_cache_size = self.config["cache_max_size_mb"] * 1024 * 1024
            
            if cache_size > max_cache_size:
                # Delete oldest files until under limit
                files_by_age = []
                for root, dirs, files in os.walk(cache_dir):
                    for file in files:
                        filepath = os.path.join(root, file)
                        mtime = os.path.getmtime(filepath)
                        size = os.path.getsize(filepath)
                        files_by_age.append((filepath, mtime, size))
                
                # Sort by age (oldest first)
                files_by_age.sort(key=lambda x: x[1])
                
                current_size = cache_size
                for filepath, mtime, size in files_by_age:
                    if current_size <= max_cache_size:
                        break
                        
                    try:
                        os.remove(filepath)
                        current_size -= size
                        savings += size
                        files_deleted += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to delete cache file {filepath}: {e}")
        
        return {
            "files_deleted": files_deleted,
            "bytes_saved": savings,
            "mb_saved": savings / (1024 * 1024)
        }
        
    def _archive_old_files(self) -> Dict:
        """Archive old files that are not frequently accessed"""
        self.logger.info("📦 Archiving old files...")
        
        # For now, just identify candidates - actual archiving would move to cheaper storage
        files_archived = 0
        potential_savings = 0
        
        archive_candidates = []
        
        for root, dirs, files in os.walk("/app/data"):
            for file in files:
                filepath = os.path.join(root, file)
                file_age_days = self._get_file_age_days(filepath)
                
                if file_age_days > self.config["archive_after_days"]:
                    file_size = os.path.getsize(filepath)
                    archive_candidates.append((filepath, file_size))
                    files_archived += 1
                    potential_savings += file_size
        
        # Log archive candidates (don't actually move in this implementation)
        if archive_candidates:
            self.logger.info(f"Found {files_archived} files eligible for archiving")
            
        return {
            "files_identified": files_archived,
            "bytes_identified": potential_savings,
            "mb_identified": potential_savings / (1024 * 1024),
            "note": "Archive candidates identified - implement S3 lifecycle for actual archiving"
        }
        
    def _is_log_file(self, filename: str) -> bool:
        """Check if file is a log file"""
        log_extensions = ['.log', '.out', '.err']
        log_keywords = ['log', 'audit', 'trace', 'debug']
        
        filename_lower = filename.lower()
        return (any(filename_lower.endswith(ext) for ext in log_extensions) or
                any(keyword in filename_lower for keyword in log_keywords))
                
    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Simple pattern matching for file cleanup"""
        if pattern.startswith('*'):
            return filename.endswith(pattern[1:])
        elif pattern.endswith('*'):
            return filename.startswith(pattern[:-1])
        else:
            return filename == pattern
            
    def _get_file_age_days(self, filepath: str) -> float:
        """Get file age in days"""
        try:
            mtime = os.path.getmtime(filepath)
            age_seconds = datetime.now().timestamp() - mtime
            return age_seconds / (24 * 3600)
        except:
            return 0
            
    def _get_file_age_hours(self, filepath: str) -> float:
        """Get file age in hours"""
        try:
            mtime = os.path.getmtime(filepath)
            age_seconds = datetime.now().timestamp() - mtime
            return age_seconds / 3600
        except:
            return 0
            
    def _get_directory_size(self, directory: str) -> int:
        """Get total size of directory"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except:
                        pass
        except:
            pass
        return total_size
        
    def _get_storage_usage(self, path: str) -> int:
        """Get storage usage for path"""
        return self._get_directory_size(path)
        
    def _calculate_cost_savings(self, bytes_saved: int) -> float:
        """Calculate estimated monthly cost savings"""
        # AWS S3 standard storage: ~$0.023 per GB per month
        gb_saved = bytes_saved / (1024 * 1024 * 1024)
        monthly_savings = gb_saved * 0.023
        return monthly_savings
        
    def generate_optimization_report(self) -> Dict:
        """Generate comprehensive optimization report"""
        if not self.optimization_results:
            return {"error": "No optimization results available"}
            
        latest_result = self.optimization_results[-1]
        
        report = {
            "optimization_summary": {
                "timestamp": latest_result["timestamp"],
                "total_savings_mb": latest_result["total_savings_mb"],
                "savings_percentage": latest_result["savings_percentage"],
                "estimated_monthly_cost_savings": latest_result["estimated_monthly_cost_savings"],
                "optimization_duration": latest_result["duration_seconds"]
            },
            "detailed_results": latest_result["optimizations"],
            "recommendations": self._generate_recommendations(),
            "next_optimization": self._schedule_next_optimization()
        }
        
        return report
        
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        if not self.optimization_results:
            return recommendations
            
        latest = self.optimization_results[-1]
        
        # Log file recommendations
        log_savings = latest["optimizations"]["log_files"]["mb_saved"]
        if log_savings > 10:
            recommendations.append("Implement automated log rotation to prevent large log accumulation")
            
        # Database recommendations
        db_savings = latest["optimizations"]["databases"]["mb_saved"]
        if db_savings > 5:
            recommendations.append("Schedule regular database maintenance and optimization")
            
        # Cache recommendations
        cache_savings = latest["optimizations"]["caches"]["mb_saved"]
        if cache_savings > 1:
            recommendations.append("Implement cache size limits and automatic cleanup")
            
        # Archive recommendations
        archive_mb = latest["optimizations"]["archives"]["mb_identified"]
        if archive_mb > 50:
            recommendations.append("Implement S3 lifecycle policies for automatic archiving")
            
        return recommendations
        
    def _schedule_next_optimization(self) -> str:
        """Determine when next optimization should run"""
        # Run daily for log cleanup, weekly for full optimization
        next_cleanup = datetime.now() + timedelta(days=1)
        next_full = datetime.now() + timedelta(days=7)
        
        return {
            "next_log_cleanup": next_cleanup.isoformat(),
            "next_full_optimization": next_full.isoformat()
        }


def main():
    """Run storage optimization"""
    print("🚀 Storage Optimization Starting...")
    print("=" * 50)
    
    optimizer = StorageOptimizer()
    
    try:
        # Run optimization
        results = optimizer.optimize_storage()
        
        # Generate report
        report = optimizer.generate_optimization_report()
        
        # Display results
        print("\n📊 OPTIMIZATION RESULTS:")
        print(f"   💾 Space saved: {results['total_savings_mb']:.1f}MB")
        print(f"   📈 Space reduction: {results['savings_percentage']:.1f}%")
        print(f"   💰 Estimated monthly savings: ${results['estimated_monthly_cost_savings']:.2f}")
        print(f"   ⏱️ Duration: {results['duration_seconds']:.1f} seconds")
        
        print("\n📋 OPTIMIZATION BREAKDOWN:")
        for category, data in results['optimizations'].items():
            if 'mb_saved' in data and data['mb_saved'] > 0:
                print(f"   {category}: {data['mb_saved']:.1f}MB saved")
                
        print("\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"   {i}. {rec}")
            
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"/app/data/storage_optimization_report_{timestamp}.json"
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n📄 Report saved: {report_file}")
        except Exception as e:
            print(f"\n⚠️ Could not save report: {e}")
            
        print("\n✅ Storage optimization completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Storage optimization failed: {e}")
        raise


if __name__ == "__main__":
    main()