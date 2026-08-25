export default function MovieCard({ title }) {
return (
<div className="bg-gray-800 p-3 rounded-xl shadow-md hover:scale-105 transition">
<p className="text-white text-sm font-light">{title}</p>
</div>
);
}